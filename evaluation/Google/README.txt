Query statistics and leakage attack accuracies of your GMail/Drive instances

########################################################################################################################

We are researching how leakage from encrypted search can be used to attack the system.
The success of known attacks highly depends on the user's behavior, but evaluations have only been performed with synthetically modeled user behavior so far.
The purpose of this evaluation is to see how the attacks work with *real users' behavior* by running relevant statistics and the attacks on your data on your machine.
With this information, we will see how exposed encrypted search algorithms are to attacks that so far are seen as theoretical.
The information does not contain any plain text, will not be passed on to other parties, and will only be used for research purposes.

########################################################################################################################

We only gather frequency statistics and attack accuracies that contain no keywords from your data.

More specifically, our scripts will only compute the following information about your GMail and or Drive data *locally on your machine*:

- Query parameters: How many queries you issued and how many distinct keywords they contain.

- Data parameters: How many documents/mails are in your data, how many distinct keywords they contain, and how many documents your queries yield on average.

- Query distribution: Power law exponent as well as plots showing the *frequencies* of the words occuring in your queries.

- Selectivity distribution: Power law exponent as well as plots showing the *frequencies* of the words occuring in your data.

- QuerySelectivity distribution: Power law exponent as well as plots showing the *frequencies* of the queries occuring in your documents.

- Attack accuracy percentages of running leakage attacks on your data.

None of this contains the keywords of your data; it just consists of frequency information of word occurrences as well as the dimension of your data. The attack accuracies contain no information of your data and just show how accurate the attacks are for your data. The attacks do not uncover any new information, and just the fact if they match the original data is reported.

########################################################################################################################

You can execute the scripts by doing the following:

1. Go to 'https://takeout.google.com/' and create the following 4 separate takeouts. If the corresponding takeout is not available to you due to no usage or due to your privacy settings, you can just skip them. Make sure your Google Language Setting (in the personal settings of your Google account) is English beforehand:

1.1. Deselect all, then select "Mail". Once the export is done (you will receive an email), export the .mbox file(s) to 'data_sources/Mail/'

1.2. Deselect all, then select "My Activity". Press the "All activity data included" button and de-select every option except for "GMail". Press the "Multiple formats" button and change "Activity records" from "HTML" to "JSON". Export the JSON activity data into the 'data_sources/MailLog/' folder.

1.3. Deselect all, then select "Drive". Once the export is done (you will receive an email), export the directory to 'data_sources/Drive/'

1.4. Deselect all, then select "My Activity". Press the "All activity data included" button and de-select every option except for "Drive". Press the "Multiple formats" button and change "Activity records" from "HTML" to "JSON". Export the JSON activity data into the 'data_sources/DriveLog/' folder.


2. Installation

2.1. LEAKER (in LEAKER directory):
pip3 install -e .

2.2. LEAKER framework dependencies (in LEAKER directory):

pip3 install -r requirements.txt

3. Index your data. This might take up to an hour and some warnings may pop up because empty documents/unreadable lines may be skipped:

python3 index_google.py


4. Load your GMail data and create the above statistics. This might also take up to an hour and some RAM if you have a lot of data. Don't run while you need your machine, e.g., in a meeting.

python3 eval_google_stats.py


5. Run Attacks. This might take some time (up to multiple hours) and RAM, don't run while you need your machine, e.g., in a meeting. Roughly, the script loads your entire data into memory multiple times.

python3 eval_google_attacks.py

- If you suffer from memory issues, please use the following flag that will increase the time but decrease memory:

python3 eval_google_attacks.py --low-memory

- If you have more memory available (~6x the data), please use the following flag:

python3 eval_google_attacks.py --high-memory


########################################################################################################################

The dimension of your data are contained in the logs: 'google_stats.log' and 'google_attack.log'
The plots of the data and attacks are stored in 'data/figures'. "*.png" files are the plot pngs, while "*.tikz" are encodings of the datapoints.
Please carefully review all of these files and, if you are ok with sharing them, send it to us (treiber@encrypto.cs.tu-darmstadt.de) with a written permission that this data may be used in a publication.
We will not acknowledge your name in any way and only use a pseudonym like 'GMail User 3'.

########################################################################################################################

For questions or if issues arise, don't hesitate to contact (treiber@encrypto.cs.tu-darmstadt.de).
