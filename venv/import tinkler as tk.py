import tkinter as tk
import re
#root object for the window
root = tk.Tk()
#attribute title and geometry
root.title("Email Splicer")
root.geometry("1600x1200")

#function to splice emails after button clicked
def splice_emails():
    emails = email_input.get(1.0, tk.END)
    #temp_label = tk.Label(root, text = emails)
    #temp_label.pack(pady=10)

    
    all_emails = []
    all_emails = re.findall(r'\S+@\S+', emails)
    print(all_emails)

    temp_email = ""
    temp_username = []
    temp_domain = []
    usernames = []
    domains = []
    temp = False
    temp_string = ""
    for i in range(len(all_emails)):
        temp_email = list(all_emails[i])
        print(temp_email)
        for j in range(len(temp_email)):
            if temp:
                temp_domain.append(temp_email[j])
            else:
                temp_username.append(temp_email[j])
            if temp_email[j]=='@':
                temp = True
            print(j)
        temp_username.pop()
        temp_string = ""
        for j in range(len(temp_username)):
            temp_string += temp_username[j]
        usernames.append(temp_string)
        temp_string = ""
        for j in range(len(temp_domain)):
            temp_string += temp_domain[j]
        domains.append(temp_string)
        temp = False
    print(usernames)
    print(domains)
    username_list = tk.Label(root, text = "Usernames: " + str(usernames))
    username_list.pack(pady=10)
    domain_list = tk.Label(root, text = "Domains: " + str(domains))
    domain_list.pack(pady=10)


#create top label
label = tk.Label(root, text = "Enter your list of emails:")
label.pack(pady=10)

#create text box
email_input = tk.Text(root, height=2, width=30)
email_input.pack(pady=5)

#create button to splice emails
splice_button = tk.Button(root, text="Splice Emails", command=splice_emails)
splice_button.pack(pady=10)


#to keep window open
root.mainloop()