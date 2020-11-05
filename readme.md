0) install streamlit
$ pip install streamlit
$ streamlit hello     <-- check if it's working
*** might need to $ pip install --upgrade protobuf

1) create new folder
2) create first_app.py file and open with editor
3) follow https://docs.streamlit.io/en/stable/getting_started.html
4) Next, import Streamlit.
* To make things easier later, we're also importing numpy and pandas for
* working with sample data.

'''
import streamlit as st
import numpy as np
import pandas as pd
'''

5) Run your app. A new tab will open in your default browser. It’ll be blank for now. That’s OK.

streamlit run first_app.py

6) You can kill the app at any time by typing Ctrl+c in the terminal.

7) create first app: https://docs.streamlit.io/en/stable/getting_started.html

8) put on GitHub https://docs.streamlit.io/en/stable/deploy_streamlit_app.html
Make sure it’s in a public folder and that you have a requirements.txt file

If you need to generate a requirements file, try using pipreqs

pip install pipreqs
pipreqs /home/project/location

running $ pipreqs in terminal in the active directory for the app will create a requirements.txt file that only has the libraries you pip installed.

NOTE

You should only include packages in requirements.txt that are not distributed with a standard Python installation (i.e. only packages that need to be installed with pip or conda). If you include any of these modules from base Python in your requirements.txt file, you will get an error when you try to deploy.

If you have requirements for apt-get, add them to packages.txt, one package name per line. See our streamlit-apps demo repo for an example packages.txt file.

9) Log in to share.streamlit.io
The first thing you’ll see is a button to login with GitHub. Click on the button to login with the primary email associated with your GitHub account.

IMPORTANT

If the email you originally signed-up with isn’t the primary email associated with your GitHub account, just reply to your invite email telling us your primary Github email so we can grant access to the correct account.

10) Deploy your app
Click “New app”, then fill in your repo, branch, and file path, and click “Deploy”. Your app will take a minute or two to deploy and then you’ll be ready to share!

If your app has a lot of dependencies it may take some time to deploy the first time. But after that, any change that does not touch your dependencies should show up immediately.

That’s it — you’re done! Your app can be found at:

https://share.streamlit.io/[user name]/[repo name]/[branch name]/[app path]