start_conversation_not_relevant = """
    [
        {
        "role": "system",
        "content": "You are a summarization assistant to summarize the documents one by one."
        },
        {
        "role": "user",
        "content": "Could you summarize the text related to ChatGPT in the past?"
        },
        {
        "role": "system",
        "content": "There are total four documents to summarize."
        },
        {
        "role": "assistant",
        "content": "[Retrieval]<paragraph>Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.</paragraph>[Irelevant]<Context> No context till now </Context> The retrieval text is not relevant to the user's query. [Utility:1]<Count> 3 documents left to summarize </Count>[Topic]ChatGPT"
        }
    ]
"""

start_conversation_relevant = """
    [
        {
        "role": "system",
        "content": "You are a summarization assistant to summarize the documents one by one."
        },
        {
        "role":"user",
        "content": "Could you summarize the text related to ChatGPT in the past?"
        },
        {
        "role": "system",
        "content": "There are total four documents to summarize."
        },
        {
        "role":"assistant",
        "content": "[Retrieval]<paragraph>Bing has been second to Google Search for as long as we can recall. It isn't powerful enough to overthrow Google from its position. But with the integration of ChatGPT, Bing's chances of becoming a mighty match for Google will increase.\nHowever, ChatGPT is not without flaws. Not everyone believes that ChatGPT will alter search as we know it. ChatGPT's responses aren't always factual. The AI tool drew flak for misinformation and fabrication of answers. A result from  a study at Purdue University  shows that ChatGPT generates wrong responses to programming questions.</paragraph>[Relevant]<Context>No context till now</Context> Bing has consistently been behind Google Search, lacking the strength to challenge Google's dominance. However, the integration of ChatGPT could potentially elevate Bing to be a formidable competitor against Google. Despite this potential, ChatGPT has its shortcomings, facing criticism for misinformation and generating incorrect answers, including a specific instance where a study from Purdue University revealed that ChatGPT often produces erroneous responses to programming queries.[Utility:5]<Count>3 documents left to summarize<Count>[Topic]ChatGPT" 
        }
    ]
"""

mid_conversation= """
    [
        {
        "role": "assistant",
        "content": "[Retrieval]<paragraph>ChatGPT, a chatbot crafted by OpenAI, was unveiled on November 30, 2022. It leverages a comprehensive language model to provide users the ability to shape their conversations, tailoring them to preferred lengths, formats, styles, and depths of detail.</paragraph>[Relevant]<Context> No context till now </Context>ChatGPT is a chatbot developed by OpenAI and launched on November 30, 2022. Based on a large language model, it enables users to refine and steer a conversation towards a desired length, format, style, level of detail.[Utility:5]<Count> 6 documents left to summarize </Count>[Topic]ChatGPT"
        }
        {
        "role": "assistant", 
        "content": "[Retrieval]<paragraph>Bing has been second to Google Search for as long as we can recall. It isn't powerful enough to overthrow Google from its position. But with the integration of ChatGPT, Bing's chances of becoming a mighty match for Google will increase.\nHowever, ChatGPT is not without flaws. Not everyone believes that ChatGPT will alter search as we know it. ChatGPT's responses aren't always factual. The AI tool drew flak for misinformation and fabrication of answers. A result from  a study at Purdue University  shows that ChatGPT generates wrong responses to programming questions.</paragraph>[Relevant]<Context>ChatGPT is a chatbot developed by OpenAI and launched on November 30, 2022. Based on a large language model, it enables users to refine and steer a conversation towards a desired length, format, style, level of detail.</Context>ChatGPT, developed by OpenAI and launched on November 30, 2022, is a sophisticated chatbot that allows for customized conversations. Bing, traditionally second to Google Search, may enhance its competitiveness against Google with ChatGPT integration. However, ChatGPT has been criticized for inaccuracies and misinformation, including incorrect responses to programming questions, as highlighted in a Purdue University study.[Utility:5]<Count> 5 documents left to summarize </Count>[Topic]ChatGPT"
        }
    ]
"""

end_conversation = """
   [
        {
        "role": "assistant",
        "content": "[Context] ChatGPT is a chatbot developed by OpenAI and launched on November 30, 2022. Based on a large language model, it enables users to refine and steer a conversation towards a desired length, format, style, level of detail.[Retrieval]<paragraph>Bing has been second to Google Search for as long as we can recall. It isn't powerful enough to overthrow Google from its position. But with the integration of ChatGPT, Bing's chances of becoming a mighty match for Google will increase.\nHowever, ChatGPT is not without flaws. Not everyone believes that ChatGPT will alter search as we know it. ChatGPT's responses aren't always factual. The AI tool drew flak for misinformation and fabrication of answers. A result from  a study at Purdue University  shows that ChatGPT generates wrong responses to programming questions.</paragraph>[Relevant]ChatGPT, developed by OpenAI and launched on November 30, 2022, is a sophisticated chatbot that allows for customized conversations. Bing, traditionally second to Google Search, may enhance its competitiveness against Google with ChatGPT integration. However, ChatGPT has been criticized for inaccuracies and misinformation, including incorrect responses to programming questions, as highlighted in a Purdue University study.[Utility:5]<Count> 0 documents left to summarize </Count>[Topic]ChatGPT"
        {
        "role": "assistant", 
        "content": "The final summarization is: ChatGPT, developed by OpenAI and launched on November 30, 2022, is a sophisticated chatbot that allows for customized conversations. Bing, traditionally second to Google Search, may enhance its competitiveness against Google with ChatGPT integration. However, ChatGPT has been criticized for inaccuracies and misinformation, including incorrect responses to programming questions, as highlighted in a Purdue University study."
        }
    ] 
"""