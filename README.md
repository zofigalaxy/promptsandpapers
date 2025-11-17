# Prompts & Papers

This is an LLM-based project developed to help researchers tackle the daily inflow of scientific papers. You describe your research interests, and your prompt gets fed to the LLM, which finds the right papers for you and sends them to your inbox regularly. This approach works very well for nuanced and niche fields, where a simple keyword search or embedding-based approaches don't classify papers with enough detail. No need to install anything; everything is handled via the interface at [promptsandpapers.com](https://promptsandpapers.com).

### Features
- Daily scraping of recent arXiv papers. Instead of using arXiv's API, we scrape papers exactly on the day they appear on the website, so you're always up-to-date.
- An LLM (right now it's GPT-4o) that classifies papers as relevant vs. not relevant based on your unique research interests. You don't need to have a huge library of papers to teach the model your interests; instead, simply describe your research interests in a prompt. You can be specific and nuanced; there's no need for vague keywords.
- A daily or weekly newsletter with the relevant papers. The selected papers include an AI-generated summary (currently using GPT-4o-mini) and a short reasoning that explains why the model thinks the paper is relevant to you. This can help you understand how the model makes the decision and refine your prompt.
- Lets you vote on whether you find a paper relevant or not. Your votes are then used to further refine your classification and make new suggestions on how to improve your prompts.
- An interface where you can see your papers, prompts, and statistics showing how many papers you receive.

### Future Improvements
This project is still evolving, and some functionalities are limited. In the future we will experiment with more LLMs, add more options and analytics, and make more sophisticated paper classifications based on the user's feedback. We welcome any comments and suggestions at contact@promptsandpapers.com!
