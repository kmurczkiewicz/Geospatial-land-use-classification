""" Helper printing functions HTML and CSS, timers etc."""
from IPython.display import Markdown, display


def print_title(string):
    string = "<div style='text-align: center;'>" + \
    "<span style='color:{}; font-size:{}; text-align:{}'>{}</span>".format("white", "24px", "center", string) + \
    "</div>"
    display(Markdown("**" + string + "**"))

    
def print_subtitle(string):
    string = "<span style='color:{}; font-size:{}'>{}</span>".format("white", "18px", string)
    white_spaces = " " * 10
    display(Markdown("**" + string + "**"))
    
    
def print_variable(string):
    string = "<span style='color:{}'>{}</span>".format("yellow", string)
    display(Markdown("**" + string + "**"))

    
def print_border():
    border = "-----------------------------------------------------------" * 3
    borderstr = "<span style='color:{}'>**{}**</span>".format("white", border)
    display(Markdown("**" + borderstr + "**"))
    
    
def print_dict(dict_to_print, spaces = ""):
    for key, value in dict_to_print.items():
        if type(value) is dict:
            spaces = spaces + " "
            print_dict(value)
        else:
            print(f"{spaces}{key} : {value}")
