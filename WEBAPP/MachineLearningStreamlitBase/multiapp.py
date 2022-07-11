"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True)
        # st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
        # app = st.sidebar.radio(
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        st.write("""<style>font-size:100px !important;</style>""", unsafe_allow_html=True)
        st.markdown(
        """<style>
        .boxBorder1 {
            outline-offset: 5px;
            font-size:20px;
        }</style>
        """, unsafe_allow_html=True) 
        # st.markdown('<div class="boxBorder1"><font color="black">Click the button.</font></div>', unsafe_allow_html=True)

        # import streamlit as st
        # st.write(self.apps)

        # st.markdown(
        #     # '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        #     '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css">',
        #     unsafe_allow_html=True,
        # )
        # query_params = st.experimental_get_query_params()
        # tabs = [self.apps[0]['title'], self.apps[1]['title'], self.apps[2]['title'], self.apps[3]['title']]
        # if "tab" in query_params:
        #     active_tab = query_params["tab"][0]
        # else:
        #     active_tab = tabs[0]# "Home"

        # if active_tab not in tabs:
        #     st.experimental_set_query_params(tab=tabs[0])
        #     active_tab = tabs[0] # "Home"
        # # # <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>

        # li_items = "".join(
        #     f"""
        #     <li class="nav-item">
        #         <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
        #     </li>
        #     """
        #     for t in tabs
        # )
        # tabs_html = f"""
        #     <ul class="nav nav-tabs">
        #     {li_items}
        #     </ul>
        # """
        # st.write(f'## {active_tab}')
        # st.markdown(tabs_html, unsafe_allow_html=True)
        # st.markdown("<br>", unsafe_allow_html=True)
        # app = self.apps[tabs.index(active_tab)]
        # if active_tab == tabs[0]:
        #     app = self.apps[0]
        #     st.write("Welcome to my lovely page!")
        #     st.write("Feel free to play with this ephemeral slider!")
        #     st.slider(
        #         "Does this get preserved? You bet it doesn't!",
        #         min_value=0,
        #         max_value=100,
        #         value=50,
        #     )
        # elif active_tab == "About":
        #     st.write("This page was created as a hacky demo of tabs")
        # elif active_tab == "Contact":
        #     st.write("If you'd like to contact me, then please don't.")
        # else:
        #     st.error("Something has gone terribly wrong.")

        from st_btn_select import st_btn_select

        # page = st_btn_select(
        #     # The different pages
        #     ('home', 'about', 'docs', 'playground'),
        #     # Enable navbar
        #     nav=True,
        #     # You can pass a formatting function. Here we capitalize the options
        #     format_func=lambda name: name.capitalize(),
        # )

        app = st_btn_select(
            # The different pages
            self.apps,
            # Enable navbar
            # nav=True,
            # You can pass a formatting function. Here we capitalize the options
            format_func=lambda app: '{}'.format(app['title']),
        )

        # Display the right things according to the page
        # if page == 'home':
        #     st.write('HOMEPAGE')





        # app = st.radio(
        #     '',
        #     self.apps,
        #     format_func=lambda app: '{}'.format(app['title']))
        #     # format_func=lambda app: '<p class="big-font">{} !!</p>'.format(app['title']))

        app['function']()