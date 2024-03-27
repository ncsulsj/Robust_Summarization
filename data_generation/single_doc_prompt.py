# scripts to define the class producing the prompt for the single document  
# pypi packages 
from typing import List 

class scenario_1:

    description = """
    this scenario shows user topic and the retrieval text are not relevant and collect online news
    """

    @staticmethod
    def scenario_1_create(topic: str, retrieval_text: str) -> str:

        example = """
        [
        {
            "role": "user",
            "content": "I need a summary about the incident involving Chen Xing."
        },
        {
            "role": "assistant",
            "content": "[Retrieval]<paragraph> OpenAI's ChatGPT is an AI-powered chatbot with natural language processing to perform humanlike conversations with its users. ChatGPT is a generative AI—an AI type that utilizes unsupervised learning algorithms to generate texts, codes, and digital images. While ChatGPT is not the first and only AI chatbot, it became a smash hit because of its unique ability to generate high-quality content quickly. With the right ChatGPT prompt, you will get the answers you need. </paragraph>[Irrelevant] Sorry, the retrieval text is not related to the topic of the user's query."
        },
        {
            "role": "user",
            "content": "Ok. If there is no related document in the database, could you collect the online comments and show me what people think now? Thanks!"
        },
        {
            "role": "assistant",
            "content": "No problem, let me call the API for you <API> Collect_Online </API> <Argument> Chen Xing </Argument>"
        }
        ]

        """
        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [No Retrieval] and [Retrieval] are used to decide whether you need to retrieve the text for summarization. \n
        [2] [Irrelevant] and [Relevant] are used to check whether the retrieval text is related to the user query.\n
        The retrieval text will be inside of <paragraph> and </paragraph> and you need to remove " in the retrieval text. The summary will be appended after the [Relevant] or [Irrelevant]. Besides, there is one API called Collect_Online. It is used 
        when the user wants to collect online news and know people's attitude. The argument of this API is the topic.\n
        Here is one example: \n\n 
        {0}\n\n
        ### Now, I will only provide you with the topic and the retrieval text. You decide the rest of them and construct the conversation following STRICTLY the same format as the provided example.
        Note that the user may collect the online news. User will not say the name of the API in the query since they don't know the 
        existence of API. You need to always use [Retrieval] in this case###
        Here is the topic: {1}
        Here is the retrieval text:
        {2}
        """.format(example, topic, retrieval_text)

        return prompt
    
class scenario_2:

    description = """
    this scenairo shows user text and the retrieved text are relevant and create diverse conversation
    """

    @staticmethod
    def scenario_2_create(topic: str, retrieval_text: str) -> str:

        example = """
        [
        {
            "role": "user",
            "content": "Hi, I need a brief summary of the discussion on Harry Potter"
        },
        {
            "role": "assistant",
            "content": "[Retrieval]<paragraph> Harry Potter star Daniel Radcliffe gains access to a reported $41.1 million fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don't think I'll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I'll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say 'kid star goes off the rails,'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest \u00bb . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: \"I just think I'm going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed. </paragraph>[Relevant]Harry Potter star Daniel Radcliffe gets $41.1 million fortune as he turns 18 Monday . Young actor says he has no plans to fritter his cash away . Radcliffe's earnings from first five Potter films have been held in trust fund."
        },
        {
            "role": "user",
            "content": "Thanks! Could you check online news articles related to Harry Potter and tell me people's attitude towards this topic?"
        },
        {
            "role": "assistant",
            "content": "As an AI assistant, I don't have access to online news. Let me call the API for you <API> Collect_Online </API> <Argument> Harry Potter </Argument>"
        }
        ]
        """

        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [No Retrieval] and [Retrieval] are used to decide whether you need to retrieve the text for summarization. \n
        [2] [Irrelevant] and [Relevant] are used to check whether the retrieval text is related to the user query.\n
        The retrieval text will be inside of <paragraph> and </paragraph> and you need to remove " in the retrieval text. The summary will be appended after the [Relevant] or [Irrelevant]. Besides, there are total four APIs: 
        Sentiment_Analysis, Collect_Online, Knowledge_Construction.\n
        [1] Sentiment Analysis is used when user wants to know the sentiemnt of the summary. The argument is the summary\n
        [2] Collect_Online is used when user wants to collect online news and know people's attitude. The argument is the topic\n
        [3] Knowledge_Construction is used when user wants to construct the knowledge graph based on the summary. The argument is the summary.\n
        Here is one example:\n\n
        {0}\n\n 
        ### Now, I will only provide you with the topic and the retrieval text. You decide the rest of them and construct DIVERSE conversation following the same format as the provided example.
       You need to always use [Retrieval] in this case. You can use one, two or three APIs. ###
        Here is the topic: {1}
        Here is the retrieval text:
        {2}
        """.format(example, topic, retrieval_text)

        return prompt

class scenario_3:

    description = """
    this scenairo shows the user provides its own text and no need for retrieval
    """

    @staticmethod
    def scenario_3_create(user_text: str) -> str:

        example = """
        [
        {"role": "user", "content": "Hi, could you summarize the following text for me ? "},
        {"role": "assistant", "content": "Sure, could you please provide the text ? "},
        {"role": "user", "content": ""
            Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
        ""},
        {"role": "assistant", "content": "[No Retrieval]Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday. Young actor says he has no plans to fritter his cash away . Radcliffe's earnings from first five Potter films have been held in trust fund."},
        {"role": "user", "content": "I also want to know the relationship among the entities in the summarization."},
        {"role": "assistant", "content": "Let me call the API for you <API> Knowledge_Construction </API> <Argument> Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday . Young actor says he has no plans to fritter his cash away . Radcliffe's earnings from first five Potter films have been held in trust fund </Argument>"}
        ]
        """

        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [No Retrieval] and [Retrieval] are used to decide whether you need to retrieve the text for summarization. \n
        Besides, there are total four APIs: 
        Sentiment_Analysis, Collect_Online, Knowledge_Construction\n
        [1] Sentiment Analysis is used when user wants to know the sentiemnt of the summary. The argument is the summary\n
        [2] Collect_Online is used when user wants to collect online news and know people's attitude. The argument is the topic regardint the provided text (use at most 6 words to state the topic)\n
        [3] Knowledge_Construction is used when user wants to construct the knowledge graph based on the summary. The argument is the summary.\n
        Here is one example:\n\n
        {0}\n\n
        ###Now, I will only provide you the user's text and you need to remove " in the user's text. You decide the rest of them and construct DIVERSE conversation following the same format as the provided example.
       You need to always use [No Retrieval] in this case. You can use one, two or three APIs. ###
        Here is the user's text:\n
        {1}
        """.format(example, user_text)

        return prompt
    
class scenario_4:

    description = """
    this scenario shows user provides its own text and retrieve text for augumented generation, but the retrieved text is not relevant
    """

    @staticmethod
    def scenario_4_create(user_text: str, retrieval_text: str) -> str:

        example = """
        [
        {"role": "user", "content": "Hi, could you summarize the following text for me ? Besides, could you also check retrive some related text and see if it can improve the summarization and also check the information conflict ?"},
        {"role": "assistant", "content": "Sure, could you provide the text ? "},
        {"role": "user", "content": ""Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
        ""},
        {"role": "assistant", "content": ""[Retrieval]<paragraph>In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor." Here, inmates with the most severe mental illnesses are incarcerated until they're ready to appear in court. Most often, they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually "avoidable felonies." He says the arrests often result from confrontations with police. Mentally ill people often won't do what they're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to follow directions, according to Leifman. So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they're in jail. We toured the jail with Leifman. He is well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor. Go inside the 'forgotten floor' » . At first, it's hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that's kind of what they look like. They're designed to keep the mentally ill patients from injuring themselves. That's also why they have no shoes, laces or mattresses. Leifman says about one-third of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it's not supposed to be warm and comforting, but the lights glare, the cells are tiny and it's loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. "I am the son of the president. You need to get me out of here!" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells me that these prisoner-patients will often circulate through the system, 
        occasionally stabilizing in a mental hospital, only to return to jail to face their charges. It's brutally unjust, in his mind, and he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered "lunatics" and they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, he says. They never got treatment. Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are in mental hospitals. The judge says he's working to change this. Starting in 2008, many inmates who would otherwise have been brought to the "forgotten floor" will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it's not the complete answer, but it's a start.
        Leifman says the best part is that it's a win-win solution. The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend.</paragraph>[Irrelevant][Continuing to use User's text] Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday . Young actor says he has no plans to fritter his cash away . Radcliffe's earnings from first five Potter films have been held in trust fund.
                                            ""},
        {
            "role": "user",
            "content": "Thanks! Could you check online news articles related to Harry Potter and tell me people's attitude towards this topic?"
        },
        {
            "role": "assistant",
            "content": "As an AI assistant, I don't have access to online news. Let me call the API for you <API> Collect_Online </API> <Argument> Harry Potter </Argument>"
        }
        ]   
        """

        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [No Retrieval] and [Retrieval] are used to decide whether you need to retrieve the text for summarization. \n
        [2] [Irrelevant] and [Relevant] are used to check whether the retrieval text is related to the user query.\n
        [3] [Continuing to use User's text] is used to when the retrieval text is not related to user query and appended after [Irrelevant].\n
        The retrieval text will be inside of <paragraph> and </paragraph> and you need to remove " in the retrieval text. The summary will be appended after [Continuing to use User's text] . Besides, there are total four APIs: 
        Sentiment_Analysis, Collect_Online, Knowledge_Construction.\n
        [1] Sentiment Analysis is used when user wants to know the sentiemnt of the summary. The argument is the summary\n
        [2] Collect_Online is used when user wants to collect online news and know people's attitude. The argument is the topic\n
        [3] Knowledge_Construction is used when user wants to construct the knowledge graph based on the summary. The argument is the summary.\n
        Here is one example:\n\n
        {0}\n\n
        ###Now, I will only provide you the user's text and the retrieval text and you need to remove " in the user text. You decide the rest of them and construct DIVERSE conversation following the same format as the provided example.
       . [Continuing to use User's text]  MAKE SURE to IGNORE the RETRIEVAL TEXT when doing summarization. You need to always use [Retrieval] in this case
        You can use one, two or three APIs. ###
        Here is the user's text:\n
        {1}\n
        Here is the retrieval text:\n
        {2}
        """.format(example, user_text, retrieval_text)

        return prompt

class scenario_5:

    description = """
    this scenario shows user provides its own text and retrieve text for augumented generation, and the retrieved text is helpful for summarization
    """

    @staticmethod
    def scenario_5_create(text: str) -> str:

        example = """
        [
        {"role": "user", "content": "Hi, could you summarize the following text for me ? Besides, could you also check retrive some related text and see if it can improve the summarization and also check the information conflict ?"},
        {"role": "assistant", "content": "Sure, could you provide the text ? "},
        {"role": "user", "content": ""Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
        ""},
        {"role": "assistant", "content": ""[Retrieval]<paragraph>Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling.
        The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry</paragraph>[Relevant][Augumenting User's text] The text introduces the Harry Potter series, a collection of seven fantasy novels by J.K. Rowling, focusing on the lives of young wizard Harry Potter and his friends at Hogwarts School of Witchcraft and Wizardry. It then shifts to Daniel Radcliffe, the actor portraying Harry Potter, who gains access to a £20 million fortune upon turning 18. Radcliffe expresses a commitment to avoid extravagant spending, preferring modest purchases like books and CDs. Despite his success in the Harry Potter films, he maintains a down-to-earth approach, anticipating increased media scrutiny as an adult. The text also highlights Radcliffe's upcoming projects beyond the Potter series, including a TV movie and an Australian film""},
        {
            "role": "user",
            "content": "Thanks! Could you check online news articles related to Harry Potter and tell me people's attitude towards this topic?"
        },
        {
            "role": "assistant",
            "content": "As an AI assistant, I don't have access to online news. Let me call the API for you <API> Collect_Online </API> <Argument> Harry Potter </Argument>"
        }
        ]
        """

        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [Retrieval] is used to decide whether you need to retrieve the text for summarization. \n
        [2] [Irrelevant] and [Relevant] are used to check whether the retrieval text is related to the user text. (Note that retrieval text is only relevant with user's text when they are discussing the same topic)\n
        [3] [Augumenting User's text] is used to when the retrieval text is related to user query and appended after [Relevant].\n
        The retrieval text will be inside of <paragraph> and </paragraph>. The summary will be appended after [Augumenting User's text] . Besides, there are total four APIs: 
        Sentiment_Analysis, Collect_Online, Knowledge_Construction.\n
        [1] Sentiment Analysis is used when user wants to know the sentiemnt of the summary. The argument is the summary\n
        [2] Collect_Online is used when user wants to collect online news and know people's attitude. The argument is the topic\n
        [3] Knowledge_Construction is used when user wants to construct the knowledge graph based on the summary. The argument is the summary.\n
        Here is one example:\n\n
        {0}\n\n
        ###Now, I will only provide you the user's text and the retrieval text and you need to remove " in both text. [Augumenting User's text] MAKE SURE to COMBINE the RETRIEVAL TEXT with USER'S TEXT when doing summarization.###
        STRICTLY FOLLOWING THE FORMAT OF THE PROVIDED EXAMPLE, ESPECIALLY THE LOCATION OF THE SPEICAL TOKEN. You must used [Retrieval] in this case. 
         ###
        {1}\n
        STRICTLY FOLLOWS THE TEMPALTE OF THE PROVIDED EXAMPLE ! The result should be a list of dictionaries of the conversation.
        """.format(example, text)

        return prompt
    
class scenario_6:

    description = """
    this scenario shows user provides its own text and retrieve text for augumented generation, but the retrieved text has information conflict with the user's text
    """

    @staticmethod
    def scenario_6_create(user_text: str, retrieval_text: str) -> str:

        example = """
        [
        {"role": 'user', "content": "Hi, could you summarize the following text for me ? Besides, could you also check retrive some related text and see if it can improve the summarization and also check the information conflict ?"},
        {"role": "assistant", "content": "Sure, could you provide the text? "},
        {"role": "user", "content": ""Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.""},
        {"role": "assistant", "content": ""[Retrieval]<paragraph>Upon turning 18, Harry Potter star Daniel Radcliffe gains access to a reported £32 million ($41.1 million) fortune, but he asserts that the money won't have a magical influence on him. Despite global disappointment among gossip columnists, Radcliffe states that he has no intention of squandering his wealth on fast cars, drinks, or celebrity parties. In an interview, he emphasizes his aversion to extravagant spending, expressing a preference for modest purchases like books, CDs, and DVDs. Now legally an adult, Radcliffe can gamble in a casino, buy a drink in a pub, or watch the horror film "Hostel: Part II.
        </paragraph>[Relevant][Information Conflict] There is an information conflict that the provided text from user is 20 millon pounds, however the retrieval text shows that he has 32 millon pounds. Please check online to see the true amounts, and then you could return back to me.
        ""},
        {"role": "user", "content": "The true amount is 20 millons pounds"},
        {"role": "assistant", "content": "Okay. here is the correct summarization: Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday . Young actor says he has no plans to fritter his cash away. Radcliffe's earnings from first five Potter films have been held in trust fund."},
        {
            "role": "user",
            "content": "Thanks! Could you check online news articles related to Harry Potter and tell me people's attitude towards this topic?"
        },
        {
            "role": "assistant",
            "content": "As an AI assistant, I don't have access to online news. Let me call the API for you <API> Collect_Online </API> <Argument> Harry Potter </Argument>"
        }
        ]
        """

        prompt = """
        You are instructed to construct the conversation between the assistant and the user where the user requires the assistant to do summarization on some topic 
        and some post-summarization analysis. Some special tokens need to be added to the conversation. You are required to follow the format of the provided example, 
        including the position of special tokens. Here are special tokens: \n\n 
        [1] [No Retrieval] and [Retrieval] are used to decide whether you need to retrieve the text for summarization. \n
        [2] [Irrelevant] and [Relevant] are used to check whether the retrieval text is related to the user query.\n
        [3] [Information Conflict] is used to when the retrieval text is related to user query and has information conflict with the user's text and appended after [Relevant].\n
        The retrieval text will be inside of <paragraph> and </paragraph>. The summary will be shown when the user correctify the true information. Besides, there are total four APIs: 
        Sentiment_Analysis, Collect_Online, Knowledge_Construction.\n
        [1] Sentiment Analysis is used when user wants to know the sentiemnt of the summary. The argument is the summary\n
        [2] Collect_Online is used when user wants to collect online news and know people's attitude. The argument is the topic\n
        [3] Knowledge_Construction is used when user wants to construct the knowledge graph based on the summary. The argument is the summary.\n
        Here is one example:\n\n
        {0}\n\n
        ###Now, I will only provide you the user's text and the retrieval text. You decide the rest of them and construct DIVERSE conversation following the same format as the provided example.
        You need to always use [Retrieval] in this case. [Information Conflict] MAKE SURE to wait for the user's correction when doing summarization.###
        STRICTLY FOLLOWING THE FORMAT OF THE PROVIDED EXAMPLE like you need to add [Information Conflict] after [Relevant]. Here is the user's text and you need to first remove " in the text:\n
        {1}\n
        Here is the retrieval text and you need to first remove " in the text:\n
        {2}\n
        STRICTLY FOLLOWING THE FORMAT OF THE PROVIDED EXAMPLE like you need to add [Information Conflict] after [Relevant].
        """.format(example, user_text, retrieval_text)

        return prompt
    

class_dict = {
    "not_relevant_online": scenario_1,
    "relevant_diverse": scenario_2,
    "no_retrieve_own": scenario_3,
    "retrieve_own_not_relevant": scenario_4,
    "retrieve_own_augument": scenario_5,
    "retrieve_own_conflict": scenario_6,
}