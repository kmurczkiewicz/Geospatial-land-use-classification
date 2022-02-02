""" Helper printing functions HTML and CSS, timers etc."""
from IPython.display import Markdown, display


def print_title(string):
    """
    Function to display str as title in jupyter notebook.
    :param string: str title to be displayed
    """
    string = "<div style='text-align: center;'>" + \
             "<span style='color:{}; font-size:{}; text-align:{}'>{}</span>".format(
                 "white",
                 "24px",
                 "center",
                 string
             ) + \
             "</div>"
    display(Markdown("**" + string + "**"))


def print_subtitle(string):
    """
    Function to display str as subtitle in jupyter notebook.
    :param string: str subtitle to be displayed
    """
    string = "<span style='color:{}; font-size:{}'>{}</span>".format("white", "18px", string)
    display(Markdown("**" + string + "**"))


def print_variable(string):
    """
    Function to display str as variable in jupyter notebook.
    :param string: str variable name to be displayed
    """
    string = "<span style='color:{}'>{}</span>".format("yellow", string)
    display(Markdown("**" + string + "**"))


def print_border():
    """
    Function to display border in jupyter notebook.
    """
    border = "-----------------------------------------------------------" * 3
    border_str = "<span style='color:{}'>**{}**</span>".format("white", border)
    display(Markdown("**" + border_str + "**"))


def print_dict(dict_to_print, spaces=""):
    """
    Function to display dictionary in jupyter notebook.

    :param dict_to_print: dict object to be displayed
    :param spaces: space amount to be used for dict displaying
    """
    for key, value in dict_to_print.items():
        if type(value) is dict:
            spaces = spaces + " "
            print_dict(value)
        else:
            print(f"{spaces}{key} : {value}")
