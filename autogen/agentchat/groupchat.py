from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Union
from .agent import Agent
from .conversable_agent import ConversableAgent
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from keywords_v2 import (
    affordability_keywords,
    safety_keywords,
    equity_keywords,
    community_keywords,
    financial_keywords,
    commercial_keywords,
)

# Create an instance of the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

logger = logging.getLogger(__name__)


@dataclass
class GroupChat:
    """(In preview) A group chat class that contains the following data fields:
    - agents: a list of participating agents.
    - messages: a list of messages in the group chat.
    - max_round: the maximum number of rounds.
    - admin_name: the name of the admin agent if there is one. Default is "Admin".
        KeyBoardInterrupt will make the admin agent take over.
    - func_call_filter: whether to enforce function call filter. Default is True.
        When set to True and when a message is a function call suggestion,
        the next speaker will be chosen from an agent which contains the corresponding function name
        in its `function_map`.
    """

    agents: List[Agent]
    messages: List[Dict]
    max_round: int = 10
    admin_name: str = "Admin"
    func_call_filter: bool = True

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    def agent_by_name(self, name: str) -> Agent:
        """Returns the agent with a given name."""
        return self.agents[self.agent_names.index(name)]

    def next_agent(self, agent: Agent, agents: List[Agent]) -> Agent:
        """Return the next agent in the list."""
        if agents == self.agents:
            return agents[(self.agent_names.index(agent.name) + 1) % len(agents)]
        else:
            offset = self.agent_names.index(agent.name) + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

    def select_speaker_msg(self, agents: List[Agent]):
        """Return the message for selecting the next speaker."""
        return f"""You are in a role play game. The following roles are available:
{self._participant_roles()}.

Read the following conversation.
Then select the next role from {[agent.name for agent in agents]} to play. Only return the role."""

    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent):
        """Select the next speaker."""
        if self.func_call_filter and self.messages and "function_call" in self.messages[-1]:
            # find agents with the right function_map which contains the function name
            agents = [
                agent for agent in self.agents if agent.can_execute_function(self.messages[-1]["function_call"]["name"])
            ]
            if len(agents) == 1:
                # only one agent can execute the function
                return agents[0]
            elif not agents:
                # find all the agents with function_map
                agents = [agent for agent in self.agents if agent.function_map]
                if len(agents) == 1:
                    return agents[0]
                elif not agents:
                    raise ValueError(
                        f"No agent can execute the function {self.messages[-1]['name']}. "
                        "Please check the function_map of the agents."
                    )
        else:
            agents = self.agents
            # Warn if GroupChat is underpopulated
            n_agents = len(agents)
            if n_agents < 3:
                logger.warning(
                    f"GroupChat is underpopulated with {n_agents} agents. Direct communication would be more efficient."
                )
        selector.update_system_message(self.select_speaker_msg(agents))
        final, name = selector.generate_oai_reply(
            self.messages
            + [
                {
                    "role": "system",
                    "content": f"Read the above conversation. Then select the next role from {[agent.name for agent in agents]} to play. Only return the role.",
                }
            ]
        )
        if not final:
            # i = self._random.randint(0, len(self._agent_names) - 1)  # randomly pick an id
            return self.next_agent(last_speaker, agents)
        try:
            return self.agent_by_name(name)
        except ValueError:
            logger.warning(
                f"GroupChat select_speaker failed to resolve the next speaker's name. Speaker selection will default to the next speaker in the list. This is because the speaker selection OAI call returned:\n{name}"
            )
            return self.next_agent(last_speaker, agents)

    def _participant_roles(self):
        roles = []
        for agent in self.agents:
            if agent.system_message.strip() == "":
                logger.warning(
                    f"The agent '{agent.name}' has an empty system_message, and may not work well with GroupChat."
                )
            roles.append(f"{agent.name}: {agent.system_message}")
        return "\n".join(roles)


class GroupChatManager(ConversableAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager.",
        **kwargs,
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply(Agent, GroupChatManager.run_chat, config=groupchat, reset_config=GroupChat.reset)
        # Allow async chat if initiated using a_initiate_chat
        self.register_reply(Agent, GroupChatManager.a_run_chat, config=groupchat, reset_config=GroupChat.reset)

    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            speaker.send(reply, self, request_reply=False)
            message = self.last_message(speaker)
            # print("run_chat, message.content=", message)

            # Start NLP Anlaysis
            # Tokenize and filter the message

            words = word_tokenize(message["content"].lower())

            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            # Lemmatize and extract keywords
            lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
            affordability_keywords_extracted = [word for word in lemmatized_words if word in affordability_keywords]
            safety_keywords_extracted = [word for word in lemmatized_words if word in safety_keywords]
            equity_keywords_extracted = [word for word in lemmatized_words if word in equity_keywords]
            community_keywords_extracted = [word for word in lemmatized_words if word in community_keywords]
            financial_keywords_extracted = [word for word in lemmatized_words if word in financial_keywords]
            commercial_keywords_extracted = [word for word in lemmatized_words if word in commercial_keywords]

            # Count occurrences
            affordability_keywords_counts = {
                keyword: affordability_keywords_extracted.count(keyword)
                for keyword in set(affordability_keywords_extracted)
            }
            safety_keywords_counts = {
                keyword: safety_keywords_extracted.count(keyword) for keyword in set(safety_keywords_extracted)
            }
            equity_keywords_counts = {
                keyword: equity_keywords_extracted.count(keyword) for keyword in set(equity_keywords_extracted)
            }
            community_keywords_counts = {
                keyword: community_keywords_extracted.count(keyword) for keyword in set(community_keywords_extracted)
            }
            financial_keywords_counts = {
                keyword: financial_keywords_extracted.count(keyword) for keyword in set(financial_keywords_extracted)
            }
            commercial_keywords_counts = {
                keyword: commercial_keywords_extracted.count(keyword) for keyword in set(commercial_keywords_extracted)
            }

            # Count total occurrences
            affordability_keywords_total = len(affordability_keywords_extracted)
            safety_keywords_total = len(safety_keywords_extracted)
            equity_keywords_total = len(equity_keywords_extracted)
            community_keywords_total = len(community_keywords_extracted)
            financial_keywords_total = len(financial_keywords_extracted)
            commercial_keywords_total = len(commercial_keywords_extracted)

            sentiment = sia.polarity_scores(" ".join(filtered_words))

            print("-------------------------------------------------")
            print("--------------- Keyword Counts ------------------")
            print("Affordability = ", affordability_keywords_counts)
            print("Safety = ", safety_keywords_counts)
            print("Equity = ", equity_keywords_counts)
            print("Community = ", community_keywords_counts)
            print("Financial = ", financial_keywords_counts)
            print("Commercial = ", commercial_keywords_counts)
            print("--------------- Keyword Total ------------------")
            print("Affordability = ", affordability_keywords_total)
            print("Safety = ", safety_keywords_total)
            print("Equity = ", equity_keywords_total)
            print("Community = ", community_keywords_total)
            print("Financial = ", financial_keywords_total)
            print("Commercial = ", commercial_keywords_total)
            print("---------Sentiment Score Analysis ---------------")
            print("Sentiment Score = ", sentiment)
            print("-------------------------------------------------")
            print("-------------------------------------------------")

        return True, None

    async def a_run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ):
        """Run a group chat asynchronously."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    await self.a_send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = await speaker.a_generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            await speaker.a_send(reply, self, request_reply=False)
            message = self.last_message(speaker)
        return True, None
