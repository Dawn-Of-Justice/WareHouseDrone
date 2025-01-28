from helper import generate, extract_text_from_pdf

pdf_path = 'Uploaded_resumes\data-scientist-1559725114.pdf'
prompt = '''
You are an advanced resume screening assistant. Extract and summarize the candidate's professional experience and key skills from the following resume.
Provide the output in the following format:

Experience Summary: List the most relevant roles, durations, and key achievements.
Key Skills: Highlight the candidate's technical, interpersonal, and domain-specific skills.
Ensure the summary is concise, focused on the recruiter's perspective, and easy to scan. Use bullet points for clarity.

Input Resume:
 
+919467891831
  
mrbriit@gmail.com
  
India
  
https://www.linkedin.com/in/mrbriit/
  
https://bit.ly/bright-portfolio
 SUMMARY
 Bright Kyeremeh
 Co-founder and Data Scientist
 2+ years in Data and Research Assistantship
 2+ years in Consultancy
 5+ years in Co-founder And Startup Management
 2+ years UDEMY Instructor For Data Science
 1+ years experience as High School Mathematics Teacher
 INTERNSHIP
 Product Manager Intern (6 Months)
 Synacor
 Helped to integrate A.I. in existing products
 Improved user productivity by introducing A.I. to achieve Email Bifurcation and Email Prioritisation.
 Integrated Priority Features(Evernote, Video Conferencing, Chat) in Zimbra Email Collaboration Software.
 Extensive Market Research and competitive analysis.
 Built and fine tuned machine learning models to predict future success of existing and targeted customers.
 Technology Consultant Intern (2 Months)
 Deloitte
 Aug '20- Jan '21
 Pune, India
 May '20- Jul '20
 Pune, India
 Leveraged large datasets from various sources at Deloitte, use the skills of Data Analysis, Natural Language Processing and Visualisation to analyze the range of 
technologies currently being used across the financial services sector and take into consideration emerging technologies for implementing an online banking 
solution.
 Helped their clients in digital transformation to integrate e-banking.
 Leveraged large datasets to analyze and draw useful insights from it.
 Present a high-level overview of the costs, benefits, and considerations of implementing online banking.
 Developed a high-level implementation plan and predicted cost estimates.
 Prepared a high-level overview of Cloud Computing for digital transformation.
 Conducted analysis and recommended which applications are suitable for transitioning to Cloud.
 Go-To-Market Strategist Intern (2 Months)
 Microsoft
 Analysing the economics of a product using strategic Go-To-Market strategy to achieve a high product market fit.
 Studied and Reported Digital Media Products and Customer Behaviour Insights.
 Analyzed market trends using Python to help position Microsoft products and services.
 Drafted compelling proposals based on industry and market analysis.
 Data Scientist Intern (2 Months)
 ANZ
 Mar '20- May '20
 Pune, India
 Jan '20- Mar '20
 Pune, India
 Leverage large dataset of over 100 customers over a period of time: analyze, visualise and build regression models to help predict behaviour of future customers 
regarding purchases, recurring transactions, and salary transactions.
 Used SQL, Tableau and Python to analyze, visualise and craft a standard report to ANZ management.
 Built machine learning models to help ANZ management predict their future customer behaviour.
 PROFESSIONAL EXPERIENCE
 Data And Research Assistant (2 years)
 Accreditation Council for Business Schools and 
Programs(ACBSP)
 Jun '15- Jul '17
 Accra, Ghana
 The Accreditation Council for Business Schools and Programs, is a U.S. organisation offering accreditation services to business programs focused on teaching 
and learning.
 Product: Offers accreditation service to business programs focused on teaching and learning
 Key Role: Identify ways in which potential institutions can make digital transformations.
Supporting Role: Serves as a consultant for the business programs in the various institutes across the African region.
 Responsibilities: 
Data Mining and Data Analysis
 Assists with academic research.
 Students’ academic programs selection counselling.
 Domain research and report writing.
 Consultation services for the various business institutes.
 ENTREPRENEURSHIP
 Founder/Information Technologist(5+ years)
 Total Data Science
 Total Data Science is a platform created to help cooperate and data science enthusiasts to up-skill in the field of data science.
 Aug '20- Present
 Delhi, India
 Build and optimize Machine Learning algorithms for companies
 Founded and serves as a product manager to create AL and Machine Learning products and services that meet industry demands 
and has a market-fit.
 Serves as a data science instructor to offer contents that make learners industry ready.
 Research and oversee the creation of contents and gamification to engage learner involvement.
 Offer consultation services to clients in introducing A.I. and Machine Learning in products.
 Offer consultation services for career development.
 UDEMY Data Science Instructor (2+ years)
 10 Courses | 2000+ Global Student base
 Teaches Python, Data Analysis, Data Science courses
 Teaches Machine Learning, Deep Learning/Artificial Intelligence courses
 RESEARCH WORK AND PUBLICATIONS(2 Publications)
 Aug '20- Present
 Delhi, India
 Kyeremeh B. (2020) . Developing and Evaluating a Knowledge-Based Health Recommender System in India Using Natural 
Language Processing Techniques to Provide Personalised Educational Materials for Chronic Disease Patients. Juni Khyat National 
journal
 Kyeremeh B. (2021). Management of individualised therapy options for chronic diseases via NLP using Precision cohort analytics. 
International Journal of Scientific & Engineering Research
 EDUCATION
 MIT ADT University
 MBA Data Science (2 years)
 Studied Data Science and Analytics as well as Artificial Intelligence for helping businesses thrive in the current age of A.I.
 Top 3 Percentile of the class
 GPA: 8.35/10 
Capstone Project
 Aug '19- Jun '21
 Pune, India
 Automatic customer complaints  tickets assignment to the appropriate IT groups using natural language processing.
 Sample Concepts Learned:
 Classification, Linear and Logistic Regression, KNN, SVM, Naive Bayes, K-Means, Random Forest, Decision Trees, PCA, NN, CNN, 
Computer Vision, NLP, Chatbot, TensorFlow,
 Descriptive Statistics, Probability,Inferential Statistics, Hypothesis Testing, etc.
 University Of Cape Coast
 Bachelor Of Education(Mathematics) (4years)
 Certified as a Ministry of Ghana High School Mathematics Instructor
 Top 5 Percentile of the class
 GPA: 3.2/4.0
 Thesis Project Work
 Implementation of technology in mathematics instructional delivery across high schools.
 KEY SKILLS
 Aug '11- Nov '16
 Cape Coast, Ghana
 Data SciencePythonMachine LearningDeep LearningArtificial IntelligenceChatbot DesignNatural Language Processing
 Computer VisionStatisticsSQLExcelWeb ScrapingData Analysis   Data MiningResearch MethodologiesReport Writing
 Data Visualization 
TECHNICAL SKILLS
 Languages: Python, SQL, Excel, Tableau
 Frameworks: TensorFlow, Pandas, Numpy, Seaborn, Selenium, Beautiful Soup, Scrapy, Flask
 OS: Windows, Linux, Mac OS
 Database: MySQL
 Cloud: AWS, Google Cloud
 CERTIFICATIONS
 IBM Certified Data Scientist
 12 Months Post Graduate Program In Artificial Intelligence 
Delhi Institute Of Digital Marketing: Professional Certificate in Social Media & Digital Marketing
 FutureLearn Certificate in Research Methodology
 FutureLearn  Certificate in Developing Research Report
 FutureLearn Certificate in Writing a Research Proposal
 Certified Ministry of Education, Ghana, High School Mathematics Instructor.
 Coursera: Data Structures & Algorithm, Operating System Design.
 EXTRACURRICULAR ACTIVITY
 Served as the President of International Students' Association, MIT ADT University campus, India.
 ADDITIONAL INFORMATION
 Languages: English-Advanced Speaker and Write

STRICTLY FOLLOW THE OUTPUT TEMPLATE. DO NOT OUTPUT ANYTHING ELSE OTHER THAT THE JSON AS GIVEN BELOW.

Output:
sample output, generate output as in the below format for every resume, for example
{
    "experience": "A brief description of the person's experience.",
    "skills": ["A list of skills the person possesses."],
    "contact_info": {
        "name": "Full name of the person.",
        "email": "Email address of the person.",
        "phone": "Phone number of the person.",
        "linkedin": "LinkedIn profile URL of the person."
    },
    "score": "A numerical score representing the person's overall rating. The scoring should be done for 100."
}
'''
text = extract_text_from_pdf(pdf_path)

out = generate(prompt + text)
print(out)

