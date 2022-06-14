#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
import tkinter.filedialog
import tkinter.font
import matplotlib.pyplot as plt
import colour
import numpy as np
import multiprocessing
import cv2
import time
import os
import pafy
import tensorflow as tf
import mtcnn
import threading

from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageTk, ImageDraw, ImageFont
from functools import partial
from tkinter import ttk

IZ_ONE = {
    "최예나":"Choi_Yena", "조유리":"Jo_Yuri", "김채원":"Kim_Chaewon", 
    "안유진":"Ahn_Yujin", "김민주":"Kim_Minjoo", "장원영":"Jang_Wonyoung",
    "나코":"Yabuki_Naco", "히토미":"Honda_Hitomi", "사쿠라":"Miyawaki_Sakura", 
    "이채연":"Lee_Chaeyeon", "강혜원":"Kang_Hyewon", "권은비":"Kwon_Eunbi"
}

IZ_ONE_index_map = [
    "Ahn_Yujin", "Choi_Yena", "Honda_Hitomi", "Jang_Wonyoung",
    "Jo_Yuri", "Kang_Hyewon", "Kim_Chaewon", "Kim_Minjoo", 
    "Kwon_Eunbi", "Lee_Chaeyeon", "Miyawaki_Sakura", "Yabuki_Naco"
]

member_color_map = {
    "Choi_Yena":(252, 246, 149), "Jo_Yuri":(243, 170, 81), "Kim_Chaewon":(206, 229, 213), 
    "Ahn_Yujin":(86, 122, 206), "Kim_Minjoo":(242, 242, 242), "Jang_Wonyoung":(217, 89, 140),
    "Yabuki_Naco":(183, 211, 233), "Honda_Hitomi":(241, 195, 170), "Miyawaki_Sakura":(241, 210, 231), 
    "Lee_Chaeyeon":(167, 224, 225), "Kang_Hyewon":(219, 112, 108), "Kwon_Eunbi":(187, 176, 220)
}

member_josa_map = {
    "Choi_Yena":"를", "Jo_Yuri":"를", "Kim_Chaewon":"을", 
    "Ahn_Yujin":"을", "Kim_Minjoo":"를", "Jang_Wonyoung":"을",
    "Yabuki_Naco":"를", "Honda_Hitomi":"를", "Miyawaki_Sakura":"를", 
    "Lee_Chaeyeon":"을", "Kang_Hyewon":"을", "Kwon_Eunbi":"를"
}

member_xy_map = {
    "Choi_Yena":(76, 130), "Jo_Yuri":(302, 130), "Kim_Chaewon":(540, 130), 
    "Ahn_Yujin":(772, 130), "Kim_Minjoo":(76, 320), "Jang_Wonyoung":(302, 320),
    "Yabuki_Naco":(540, 320), "Honda_Hitomi":(772, 320), "Miyawaki_Sakura":(76, 510), 
    "Lee_Chaeyeon":(302, 510), "Kang_Hyewon":(540, 510), "Kwon_Eunbi":(772, 510)
}

reversed_IZ_ONE = {v:k for k, v in IZ_ONE.items()}

buttons_xy_map = {
    "prev":(300, 720, 150, 50), "next":(600, 720, 150, 50), "youtube_btn":(75, 500, 250, 50), 
    "intranet_btn":(375, 500, 250, 50), "localfile_btn":(675, 500, 250, 50)
}

entry_text_list = [
    "   유튜브 url을 입력하세요", "   인트라넷 주소을 입력하세요", "   옆의 열기 버튼을 눌러 업로드하세요"
]

entry_text_mode_dict = {
    "   유튜브 url을 입력하세요":"y", "   인트라넷 주소을 입력하세요":"i", "   옆의 열기 버튼을 눌러 업로드하세요":"l"
}

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error : Creating directory. " + directory)

def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

member_classification_model = tf.keras.models.load_model("member_classification_model_B7.h5")

detector = MTCNN()

confidence = 0.99999

# gui

window = tk.Tk()

window_width = 1200
window_height = 895

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

window.title("당신의 최애를 자동으로 캡쳐해드립니다!")
window.iconbitmap("icon.ico")
window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
window.resizable(False, False)

font_eg_group = tkinter.font.Font(family="Rampart One", size=100)
font_eg_program = tkinter.font.Font(family="Rampart One", size=50)
font_kr = tkinter.font.Font(family="스웨거 TTF", size=35)
font_kr_mode_btn = tkinter.font.Font(family="스웨거 TTF", size=22)

picked_member_name = "멤버를 선택하세요!"

video_url = "영상의 주소를 입력하세요!"

img = Image.open("./background_img.jpg")
img = ImageTk.PhotoImage(img)
background_img = tk.Label(window, image=img)
background_img.image = img
background_img.place(x=0, y=0, relwidth=1, relheight=1)


style = ttk.Style()

style.layout('TNotebook.Tab', [])

notebook_width=1000
notebook_height=800

notebook = ttk.Notebook(window, width=notebook_width, height=notebook_height)
notebook.place(x=(window_width/2-500), y=(window_height/2-400))

select_mode_tab = tk.Frame(window)
notebook.add(select_mode_tab, text="select_mode_tab")

select_member_tab = tk.Frame(window)
notebook.add(select_member_tab, text="select_member_tab")

select_dir_tab = tk.Frame(window)
notebook.add(select_dir_tab, text="select_dir_tab")

tab_list = [select_mode_tab, select_member_tab, select_dir_tab]

# functions
def fade(widget, smoothness=4, cnf={}, **kw):
    kw = tk._cnfmerge((cnf, kw))
    if not kw: raise ValueError("No option given, -bg, -fg, etc")
    if len(kw)>1: return [fade(widget,smoothness,{k:v}) for k,v in kw.items()][0]
    if not getattr(widget, '_after_ids', None): widget._after_ids = {}
    widget.after_cancel(widget._after_ids.get(list(kw)[0], ' '))
    c1 = tuple(map(lambda a: a/(65535), widget.winfo_rgb(widget[list(kw)[0]])))
    c2 = tuple(map(lambda a: a/(65535), widget.winfo_rgb(list(kw.values())[0])))
    colors = tuple(colour.rgb2hex(c, force_long=True)
                   for c in colour.color_scale(c1, c2, max(1, smoothness*100)))

    def worker(count=0):
        if len(colors)-1 <= count: return
        widget.config({list(kw)[0] : colors[count]})
        widget._after_ids.update( { list(kw)[0]: widget.after(
            max(1, int(smoothness/10)), worker, count+1) } )
    worker()
    
def bg_config(widget, bg, fg, event):            
    fade(widget, smoothness=5, bg=bg)

def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func

def member_btn_clicked(text):
    global picked_member_name
    picked_member_name = text
    
    select_member_tab_canvas.itemconfig(member_name_text, text=reversed_IZ_ONE[picked_member_name]+member_josa_map[picked_member_name]+" 선택하셨습니다!")
    
    font = ImageFont.truetype("fonts/gulim.ttc", 22)
    
    mt_prev_img = Image.open(f"./buttons/{picked_member_name}_not_selected_btn.png")
    mt_prev_img = ImageTk.PhotoImage(mt_prev_img)
    mt_prev_btn = tk.Button(select_member_tab, image=mt_prev_img, 
                         highlightthickness = 0, bd = 0, command=select_prev_tab)
    mt_prev_btn.image = mt_prev_img
    mt_prev_btn.place(x=notebook_width/2 - 200, y=710, width=150, height=56)
    mt_prev_btn.bind("<Enter>", partial(mode_button_hover, mt_prev_btn, 
                                       "buttons", f"{picked_member_name}_not_selected_btn", notebook_width/2 - 200 - 5, 710 - 1, 150, 56))
    mt_prev_btn.bind("<Leave>", partial(button_hover_leave, mt_prev_btn, 
                                        "buttons", f"{picked_member_name}_not_selected_btn", notebook_width/2 - 200, 710, 150, 56))
    
    try: mt_not_selected_next_btn.place_forget()
    except: pass
    
    mt_selected_next_btn_img = Image.open(f"./buttons/{picked_member_name}_selected_btn.png")
    mt_selected_next_btn_img = ImageTk.PhotoImage(mt_selected_next_btn_img)
    mt_selected_next_btn = tk.Button(select_member_tab, image=mt_selected_next_btn_img,
                                  highlightthickness = 0, bd = 0, command=select_next_tab)
    mt_selected_next_btn.image = mt_selected_next_btn_img
    mt_selected_next_btn.place(x=notebook_width/2 + 50, y=710, width=150, height=56)
    mt_selected_next_btn.bind("<Enter>", partial(mode_button_hover, mt_selected_next_btn, 
                              "buttons", f"{picked_member_name}_selected_btn", notebook_width/2  + 50 - 5, 710 - 1, 150, 56))
    mt_selected_next_btn.bind("<Leave>", partial(button_hover_leave, mt_selected_next_btn, 
                              "buttons", f"{picked_member_name}_selected_btn", notebook_width/2 + 50, 710, 150, 56))
    make_select_dir_tab()
    
def button_hover(widget, directory, name, x, y, w, h, e):
    img = Image.open(f"./{directory}/{name}.png").resize((w + 10, h + 10))
    img = ImageTk.PhotoImage(img)
    widget.config(image=img)
    widget.image = img
    widget.place(x=x, y=y, width=w+10, height=h+10)

def mode_button_hover(widget, directory, name, x, y, w, h, e):
    img = Image.open(f"./{directory}/{name}.png").resize((w + 10, h + 2))
    img = ImageTk.PhotoImage(img)
    widget.config(image=img)
    widget.image = img
    widget.place(x=x, y=y, width=w+10, height=h+2)
    
def button_hover_leave(widget, directory, name, x, y, w, h, e):
    img = Image.open(f"./{directory}/{name}.png")
    img = ImageTk.PhotoImage(img)
    widget.config(image=img)
    widget.image = img
    widget.place(x=x, y=y, width=w, height=h)
    
def make_member_img_btn(name):
    global img
    x = member_xy_map[name][0]
    y = member_xy_map[name][1]
    w = 152
    h = 150
    img = Image.open(f"./member_icon/{name}.png")
    img = ImageTk.PhotoImage(img)
    img_btn = tk.Button(select_member_tab, image=img, command=partial(member_btn_clicked, name),                         highlightthickness = 0, bd = 0, cursor="hand2", borderwidth=0)
    img_btn.image = img
    img_btn.place(x=x, y=y, width=w, height=h)
    img_btn.bind("<Button-1>", partial(bg_config, 
                select_member_tab_canvas, _from_rgb(member_color_map[name]), "white"))
    img_btn.bind("<Enter>", partial(button_hover, img_btn, "member_icon", name, x - 5, y - 5, w, h))
    img_btn.bind("<Leave>", partial(button_hover_leave, img_btn, "member_icon", name, x, y, w, h))
    
    
def make_mode_btn(name, text):
    global img
    x = buttons_xy_map[name][0]
    y = buttons_xy_map[name][1]
    w = buttons_xy_map[name][2]
    h = buttons_xy_map[name][3]
    mode = entry_text_mode_dict[text]
    
    img = Image.open(f"./buttons/{name}.png")
    img = ImageTk.PhotoImage(img)
    img_btn = tk.Button(select_mode_tab, image=img, text=text, highlightthickness = 0, 
                        bd = 0, cursor="hand2", command=partial(show_entry, mode))
    img_btn.image = img
    img_btn.place(x=x, y=y, width=w, height=h)
    img_btn.bind("<Enter>", partial(mode_button_hover, img_btn, "buttons", name, x - 5, y - 1, w, h))
    img_btn.bind("<Leave>", partial(button_hover_leave, img_btn, "buttons", name, x, y, w, h))
    
def file_search_window(widget):
    global entry_input
    filename = tkinter.filedialog.askopenfilename(initialdir='/',title="select a file",
                                        filetypes =(("Video file","*.mp4"), ("all files","*.*"))) ##################
    
    widget.delete(0, "end")
    widget.insert(0, filename)

def dir_search_window(widget):
    global entry_input
    filename = tkinter.filedialog.askdirectory(initialdir='/',title="select a directory") ##################
    print(filename)
    widget.delete(0, "end")
    widget.insert(0, filename)
    
def entry_default_text_destroy(widget, e):
    e.widget.delete(0, "end") 
    
def show_entry(mode):
    if mode == "y":
        youtube_entry.delete(0, "end")
        youtube_entry.insert(0, entry_text_list[0])
        
        youtube_entry.place(x=(notebook_width-400)/2, y = 600, width = 400, height=50)
        youtube_entry.bind("<Button-1>", partial(entry_default_text_destroy, youtube_entry))
        try:
            intranet_entry.place_forget()
            localfile_entry.place_forget()
            file_search_btn.place_forget()
        except: pass
    elif mode == "i":
        intranet_entry.delete(0, "end")
        intranet_entry.insert(0, entry_text_list[1])
        
        intranet_entry.place(x=(notebook_width-400)/2, y = 600, width = 400, height=50)
        intranet_entry.bind("<Button-1>", partial(entry_default_text_destroy, intranet_entry))
        try:
            youtube_entry.place_forget()
            localfile_entry.place_forget()
            file_search_btn.place_forget()
        except: pass
    elif mode== "l":
        localfile_entry.delete(0, "end")
        localfile_entry.insert(0, entry_text_list[2])
        
        localfile_entry.place(x=250, y = 600, width = 400, height=50)
        localfile_entry.bind("<Button-1>", partial(entry_default_text_destroy, localfile_entry))
        
        file_search_btn.place(x=700, y=600, width=50, height=50)
        try:
            youtube_entry.place_forget()
            intranet_entry.place_forget()
        except: pass
    else: print("Check parameter")
    
# def make_prev_next_btn():
#     w = 152
#     p_x = notebook_width/2 - 202
#     ns_x = notebook_width/2 + 50
#     y = 710
#     h = 58
#     prev_img = Image.open(f"./buttons/prev_btn.png")
#     prev_img = ImageTk.PhotoImage(prev_img)
#     prev_btn = tk.Button(select_mode_tab, image=prev_img, 
#                          highlightthickness = 0, bd = 0)
#     prev_btn.image = prev_img
#     prev_btn.place(x=notebook_width/2 - 202, y=710, width=152, height=58)
#     prev_btn.bind("<Enter>", partial(mode_button_hover, prev_btn, "buttons", "prev_btn", notebook_width/2 - 202 - 5, 710 - 5, 152, 58))
#     prev_btn.bind("<Leave>", partial(button_hover_leave, prev_btn, "buttons", "prev_btn", notebook_width/2 - 202, 710, 152, 58))  
    
#     not_selected_next_btn_img = Image.open(f"./buttons/not_selected_next_btn.png")
#     not_selected_next_btn_img = ImageTk.PhotoImage(not_selected_next_btn_img)
#     not_selected_next_btn = tk.Button(select_mode_tab, image=not_selected_next_btn_img, 
#                                       highlightthickness = 0, bd = 0)
#     not_selected_next_btn.image = not_selected_next_btn_img
#     not_selected_next_btn.place(x=notebook_width/2 + 50, y=710, width=152, height=58)
#     not_selected_next_btn.bind("<Enter>", partial(mode_button_hover, not_selected_next_btn, 
#                                                   "buttons", "not_selected_next_btn", notebook_width/2 + 50 - 5, 710 - 5, 152, 58))
#     not_selected_next_btn.bind("<Leave>", partial(button_hover_leave, not_selected_next_btn, 
#                                                   "buttons", "not_selected_next_btn", notebook_width/2 + 50, 710, 152, 58))  
    
#     selected_next_btn_img = Image.open(f"./buttons/selected_next_btn.png")
#     selected_next_btn_img = ImageTk.PhotoImage(selected_next_btn_img)
#     selected_next_btn = tk.Button(select_mode_tab, image=selected_next_btn_img,
#                                   highlightthickness = 0, bd = 0)
#     selected_next_btn.image = selected_next_btn_img

def select_prev_tab():
    notebook.select(tab_list[notebook.index(notebook.select()) - 1])
    
def select_next_tab():
    notebook.select(tab_list[notebook.index(notebook.select()) + 1])

def entry_modified(sv_widget, ns_btn, s_btn):
    contents = sv_widget.get()
    
    w = 150
    ns_x = (notebook_width-150)/2
    y = 710
    h = 56
    
    if not ("   " in contents or contents == ""):
        try:
            ns_btn.place_forget()
        except: pass
        
        s_btn.place(x=ns_x, y=y, width=w, height=h)
        s_btn.bind("<Enter>", partial(mode_button_hover, selected_next_btn, 
                                                  "buttons", "selected_next_btn", ns_x - 5, y - 1, w, h))
        s_btn.bind("<Leave>", partial(button_hover_leave, selected_next_btn, 
                                                  "buttons", "selected_next_btn", ns_x, y, w, h))    
    else:
        try: 
            s_btn.place_forget()
        except: pass
        
        ns_btn.place(x=ns_x, y=y, width=w, height=h)
        ns_btn.bind("<Enter>", partial(mode_button_hover, not_selected_next_btn, 
                                                      "buttons", "not_selected_next_btn", ns_x - 5, y - 1, w, h))
        ns_btn.bind("<Leave>", partial(button_hover_leave, not_selected_next_btn, 
                                                      "buttons", "not_selected_next_btn", ns_x, y, w, h))    
        
def entry_modified_trd_frame(sv_widget, ns_btn, s_btn):
    contents = sv_widget.get()
    
    w = 150
    ns_x = notebook_width/2 + 50
    y = 710
    h = 56
    
    if not ("   " in contents or contents == ""):
        try:
            ns_btn.place_forget()
        except: pass
        
        try:
            s_btn.place(x=ns_x, y=y, width=w, height=h)
            s_btn.bind("<Enter>", partial(mode_button_hover, dt_selected_activate_btn, 
                                                      "buttons", f"{picked_member_name}_selected_activate_btn", ns_x - 5, y - 1, w, h))
            s_btn.bind("<Leave>", partial(button_hover_leave, dt_selected_activate_btn, 
                                                      "buttons", f"{picked_member_name}_selected_activate_btn", ns_x, y, w, h))    
        except: pass
    else:
        try: 
            s_btn.place_forget()
        except: pass
        
        try: 
            ns_btn.place(x=ns_x, y=y, width=w, height=h)
            ns_btn.bind("<Enter>", partial(mode_button_hover, dt_not_selected_activate_btn, 
                                                          "buttons", f"{picked_member_name}_not_selected_activate_btn", ns_x - 5, y - 1, w, h))
            ns_btn.bind("<Leave>", partial(button_hover_leave, dt_not_selected_activate_btn, 
                                                          "buttons", f"{picked_member_name}_not_selected_activate_btn", ns_x, y, w, h)) 
        except: pass
        
# select_mode_tab    

background_mode_label = tk.Label(select_mode_tab, bg="white")
background_mode_label.image = img
background_mode_label.place(x=0, y=0, width=1000, height=800)

group_name_logo_label = tk.Label(select_mode_tab, font=font_eg_group, text="IZ ONE", 
                                 relief="flat", bg="white", fg=_from_rgb((236, 95, 157)))
program_name_logo_label = tk.Label(select_mode_tab, font=font_eg_program, text="FINDER", 
                                   relief="flat", bg="white", fg=_from_rgb((236, 95, 157)))
group_name_logo_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
program_name_logo_label.place(relx=0.5, rely=0.35, anchor=tk.CENTER)


not_selected_next_btn_img = Image.open(f"./buttons/not_selected_next_btn.png")
not_selected_next_btn_img = ImageTk.PhotoImage(not_selected_next_btn_img)
not_selected_next_btn = tk.Button(select_mode_tab, image=not_selected_next_btn_img, 
                                  highlightthickness = 0, bd = 0)
not_selected_next_btn.image = not_selected_next_btn_img
not_selected_next_btn.place(x=(notebook_width-150)/2, y=710, width=150, height=56)
not_selected_next_btn.bind("<Enter>", partial(mode_button_hover, not_selected_next_btn, 
                                              "buttons", "not_selected_next_btn", (notebook_width-150)/2 - 5, 710 - 1, 150, 56))
not_selected_next_btn.bind("<Leave>", partial(button_hover_leave, not_selected_next_btn, 
                                              "buttons", "not_selected_next_btn", (notebook_width-150)/2, 710, 150, 56))  

selected_next_btn_img = Image.open(f"./buttons/selected_next_btn.png")
selected_next_btn_img = ImageTk.PhotoImage(selected_next_btn_img)
selected_next_btn = tk.Button(select_mode_tab, image=selected_next_btn_img,
                              highlightthickness = 0, bd = 0, command=select_next_tab)
selected_next_btn.image = selected_next_btn_img    
        
    
youtube_entry_default = tk.StringVar()
youtube_entry_default.set(entry_text_list[0])
youtube_entry_default.trace("w", lambda name, index, sv=youtube_entry_default:entry_modified(youtube_entry_default, not_selected_next_btn, selected_next_btn))
youtube_entry = tk.Entry(select_mode_tab, textvariable=youtube_entry_default, 
                         font=font_kr_mode_btn, highlightthickness=0, bd=0,
                         background=_from_rgb((255, 0, 0))) 
youtube_entry.place(x=(notebook_width-400)/2, y = 600, width = 400, height=50)
youtube_entry.bind("<Button-1>", partial(entry_default_text_destroy, youtube_entry))

intranet_entry_default = tk.StringVar()
intranet_entry_default.set(entry_text_list[1])
intranet_entry_default.trace("w", lambda name, index, sv=intranet_entry_default:entry_modified(intranet_entry_default, not_selected_next_btn, selected_next_btn))
intranet_entry = tk.Entry(select_mode_tab, textvariable=intranet_entry_default, 
                          font=font_kr_mode_btn, highlightthickness=0, bd=0,
                          background=_from_rgb((255, 227, 117)))

localfile_entry_default = tk.StringVar()
localfile_entry_default.set(entry_text_list[2])
localfile_entry_default.trace("w", lambda name, index, sv=localfile_entry_default:entry_modified(localfile_entry_default, not_selected_next_btn, selected_next_btn))
localfile_entry = tk.Entry(select_mode_tab, textvariable=localfile_entry_default, 
                           font=font_kr_mode_btn, highlightthickness=0, bd=0,
                           background=_from_rgb((126, 190, 234)))

file_search_btn_img = Image.open("./buttons/open_filesearch_btn.png")
file_search_btn_img = ImageTk.PhotoImage(file_search_btn_img)
file_search_btn = tk.Button(select_mode_tab, text="열기", font=font_kr_mode_btn, 
                            command=partial(file_search_window, localfile_entry), 
                            highlightthickness = 0, bd = 0, cursor="hand2", 
                            image=file_search_btn_img)

file_search_btn.bind("<Enter>", partial(button_hover, file_search_btn, "buttons", "open_filesearch_btn", 695, 595, 50, 50))
file_search_btn.bind("<Leave>", partial(button_hover_leave, file_search_btn, "buttons", "open_filesearch_btn", 700, 600, 50, 50))



for name, text in zip(["youtube_btn", "intranet_btn", "localfile_btn"], entry_text_mode_dict.keys()):
    make_mode_btn(name, text)
     


# select_member_tab

# background_member_label = tk.Label(select_member_tab, bg="white")
# background_member_label.image = img
# background_member_label.place(x=0, y=0, width=1000, height=800)

# member_name_label = tk.Label(select_member_tab, text=picked_member_name, font=font_kr, relief="solid")
# member_name_label.place(x=notebook_width/2-250, y=35, width=500, height=40)

select_member_tab_canvas = tk.Canvas(select_member_tab, bg="white")
select_member_tab_canvas.place(x=0, y=0, width=window_width, height=window_height)
member_name_text = select_member_tab_canvas.create_text(500, 65, text=picked_member_name, font=font_kr, fill="black")

mt_prev_img = Image.open(f"./buttons/prev_btn.png")
mt_prev_img = ImageTk.PhotoImage(mt_prev_img)
mt_prev_btn = tk.Button(select_member_tab, image=mt_prev_img, 
                     highlightthickness = 0, bd = 0, command=select_prev_tab)
mt_prev_btn.image = mt_prev_img
mt_prev_btn.place(x=notebook_width/2 - 200, y=710, width=150, height=56)
mt_prev_btn.bind("<Enter>", partial(mode_button_hover, mt_prev_btn, "buttons", "prev_btn", notebook_width/2 - 200 - 5, 710 - 1, 150, 56))
mt_prev_btn.bind("<Leave>", partial(button_hover_leave, mt_prev_btn, "buttons", "prev_btn", notebook_width/2 - 200, 710, 150, 56))  

mt_not_selected_next_btn_img = Image.open(f"./buttons/not_selected_next_btn.png")
mt_not_selected_next_btn_img = ImageTk.PhotoImage(mt_not_selected_next_btn_img)
mt_not_selected_next_btn = tk.Button(select_member_tab, image=mt_not_selected_next_btn_img, 
                                  highlightthickness = 0, bd = 0)
mt_not_selected_next_btn.image = mt_not_selected_next_btn_img
mt_not_selected_next_btn.place(x=notebook_width/2 + 50, y=710, width=150, height=56)
mt_not_selected_next_btn.bind("<Enter>", partial(mode_button_hover, mt_not_selected_next_btn, 
                                              "buttons", "not_selected_next_btn", notebook_width/2  + 50 - 5, 710 - 1, 150, 56))
mt_not_selected_next_btn.bind("<Leave>", partial(button_hover_leave, mt_not_selected_next_btn, 
                                              "buttons", "not_selected_next_btn", notebook_width/2 + 50, 710, 150, 56))

mt_selected_next_btn_img = Image.open(f"./buttons/selected_next_btn.png")
mt_selected_next_btn_img = ImageTk.PhotoImage(mt_selected_next_btn_img)
mt_selected_next_btn = tk.Button(select_member_tab, image=mt_selected_next_btn_img,
                              highlightthickness = 0, bd = 0, command=select_next_tab)
mt_selected_next_btn.image = selected_next_btn_img

img = None
for name in IZ_ONE.values():
    make_member_img_btn(name)

# select_dir_tab

loading_canvas = None
dt_localfile_entry = None
dt_selected_activate_btn = None
def make_select_dir_tab():
    global picked_member_name, dt_selected_activate_btn, loading_canvas, dt_localfile_entry
    
    dir_tab_color = _from_rgb(member_color_map[picked_member_name])

    select_dir_tab_canvas = tk.Canvas(select_dir_tab, bg = dir_tab_color)
    select_dir_tab_canvas.place(x=0, y=0, width=window_width, height=window_height)
    dir_name_text = select_dir_tab_canvas.create_text(500, 65, text="사진들을 저장할 위치를 지정하세요!", font=font_kr, fill="black")

    dt_prev_img = Image.open(f"./buttons/{picked_member_name}_not_selected_btn.png")
    dt_prev_img = ImageTk.PhotoImage(dt_prev_img)
    dt_prev_btn = tk.Button(select_dir_tab, image=dt_prev_img, 
                         highlightthickness = 0, bd = 0, command=select_prev_tab)
    dt_prev_btn.image = dt_prev_img
    dt_prev_btn.place(x=notebook_width/2 - 200, y=710, width=150, height=56)
    dt_prev_btn.bind("<Enter>", partial(mode_button_hover, dt_prev_btn, "buttons", f"{picked_member_name}_not_selected_btn", notebook_width/2 - 200 - 5, 710 - 1, 150, 56))
    dt_prev_btn.bind("<Leave>", partial(button_hover_leave, dt_prev_btn, "buttons", f"{picked_member_name}_not_selected_btn", notebook_width/2 - 200, 710, 150, 56))  

    dt_not_selected_activate_btn_img = Image.open(f"./buttons/{picked_member_name}_not_selected_activate_btn.png")
    dt_not_selected_activate_btn_img = ImageTk.PhotoImage(dt_not_selected_activate_btn_img)
    dt_not_selected_activate_btn = tk.Button(select_dir_tab, image=dt_not_selected_activate_btn_img, 
                                        highlightthickness = 0, bd = 0)
    dt_not_selected_activate_btn.image = dt_not_selected_activate_btn_img
    dt_not_selected_activate_btn.place(x=notebook_width/2 + 50, y=710, width=150, height=56)
    dt_not_selected_activate_btn.bind("<Enter>", partial(mode_button_hover, dt_not_selected_activate_btn, 
                                                    "buttons", f"{picked_member_name}_not_selected_activate_btn", notebook_width/2  + 50 - 5, 710 - 1, 150, 56))
    dt_not_selected_activate_btn.bind("<Leave>", partial(button_hover_leave, dt_not_selected_activate_btn, 
                                                    "buttons", f"{picked_member_name}_not_selected_activate_btn", notebook_width/2 + 50, 710, 150, 56))

    dt_selected_activate_btn_img = Image.open(f"./buttons/{picked_member_name}_selected_activate_btn.png")
    dt_selected_activate_btn_img = ImageTk.PhotoImage(dt_selected_activate_btn_img)
    dt_selected_activate_btn = tk.Button(select_dir_tab, image=dt_selected_activate_btn_img,
                                    highlightthickness = 0, bd = 0, command=run_model)
    dt_selected_activate_btn.image = dt_selected_activate_btn_img


    dt_localfile_entry_default = tk.StringVar()
    dt_localfile_entry_default.set(entry_text_list[2])
    dt_localfile_entry_default.trace("w", lambda name, index, sv=localfile_entry_default:entry_modified_trd_frame(dt_localfile_entry_default, dt_not_selected_activate_btn, dt_selected_activate_btn))
    dt_localfile_entry = tk.Entry(select_dir_tab, textvariable=dt_localfile_entry_default, 
                               font=font_kr_mode_btn, highlightthickness=0, bd=0,
                               background=_from_rgb((126, 190, 234)))
    dt_localfile_entry.place(x=250, y = 400, width = 400, height=50)
    dt_localfile_entry.bind("<Button-1>", partial(entry_default_text_destroy, dt_localfile_entry))
    
    dt_file_search_btn_img = Image.open(f"./buttons/{picked_member_name}_open_filesearch_btn.png")
    dt_file_search_btn_img = ImageTk.PhotoImage(dt_file_search_btn_img)
    dt_file_search_btn = tk.Button(select_dir_tab, text="열기", font=font_kr_mode_btn, 
                                command=partial(dir_search_window, dt_localfile_entry), 
                                highlightthickness = 0, bd = 0, cursor="hand2", 
                                image=dt_file_search_btn_img)
    dt_file_search_btn.image = dt_file_search_btn_img
    dt_file_search_btn.bind("<Enter>", partial(button_hover, dt_file_search_btn, "buttons", f"{picked_member_name}_open_filesearch_btn", 695, 395, 50, 50))
    dt_file_search_btn.bind("<Leave>", partial(button_hover_leave, dt_file_search_btn, "buttons", f"{picked_member_name}_open_filesearch_btn", 700, 400, 50, 50))
    dt_file_search_btn.place(x=700, y=400, width=50, height=50)
    
    loading_canvas = tk.Canvas(select_dir_tab, bg = dir_tab_color)
    
loading_text = "사진을 캡쳐중 입니다."
loading_text_id = None
complete_text = "완료되었습니다!"
loading_chk = True

images = []

def create_rectangle(x1, y1, x2, y2, **kwargs):
    global loading_canvas
    if 'alpha' in kwargs:
        alpha = int(kwargs.pop('alpha') * 255)
        fill = kwargs.pop('fill')
        fill = window.winfo_rgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        images.append(ImageTk.PhotoImage(image))
        loading_canvas.create_image(x1, y1, image=images[-1], anchor='nw')
    loading_canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

def loading_animation():
    global loading_text, loading_canvas, complete_text
    
    if "......" in loading_text: loading_text = "사진을 캡쳐중 입니다."    
                
    if loading_chk:
        loading_canvas.itemconfig(loading_text_id, text=loading_text)
        loading_text = loading_text + "."

        window.after(1000, loading_animation)
    else:
        loading_canvas.delete("all")
        create_rectangle(0, 0, window_width, window_height, fill="white", outline="", alpha=0.6)
        loading_canvas.create_text(500, 400, text=complete_text, font=font_kr, fill="black")
    
def check_entry_text(widget):
    text = widget.get()
    if "   " in text or text == "": return False
    else: return True

def run_model():
    global video_url, picked_member_name, loading_text, loading_canvas, dt_localfile_entry, complete_text, loading_text_id

    loading_canvas.place(x=0, y=0, width=window_width, height=window_height)
    create_rectangle(0, 0, window_width, window_height, fill="white", outline="", alpha=0.6)
    loading_text_id = loading_canvas.create_text(500, 400, text=loading_text, font=font_kr, fill="black")
    loading_text = loading_text + "."  
    
    window.after(1000, loading_animation)
    
    member_index = IZ_ONE_index_map.index(picked_member_name)
    save_directory = dt_localfile_entry.get()
      
    try:
        if check_entry_text(youtube_entry):
            print("youtube")
            video_url = youtube_entry.get()
#             youtube_video_parsing(video_url, member_index, save_directory, member_classification_model)
            t = threading.Thread(target=youtube_video_parsing, args=(video_url, member_index, save_directory, member_classification_model ))
#             videoparsing.youtube_video_parsing(video_url, member_index, save_directory, member_classification_model)
        elif check_entry_text(intranet_entry):
            print("intranet")
            video_url = intranet_entry.get()
#             t = threading.Thread(target=intranet_video_parsing, args=(video_url, member_index, save_directory, member_classification_model ))
    #         videoparsing.intranet_video_parsing(video_url, member_index, save_directory, member_classification_model)
        elif check_entry_text(localfile_entry):
            print("localfile")
            video_url = localfile_entry.get()
#             local_video_parsing(video_url, member_index, save_directory, member_classification_model)
            t = threading.Thread(target=local_video_parsing, args=(video_url, member_index, save_directory, member_classification_model ))
#             videoparsing.local_video_parsing(video_url, member_index, save_directory, member_classification_model)
    except:
        complete_text = "잘못된 영상의 주소입니다. 다시 시작하십시오."
        
    t.start() 

def face_detect_and_member_classification(image, member_index, save_dir, member_classification_model):
    results = detector.detect_faces(image)

    face_list = []

    for result in results:
        x1, y1, width, height = result["box"]
        x2, y2 = x1 + width, y1 + height

        face = image[y1:y2, x1:x2]
                
        face_image = Image.fromarray(face)
        face_image = face_image.resize((224, 224))
        face_image = np.asarray(face_image)
        face_image = face_image / 255        
        face_image = np.array([face_image])

        member_predict = member_classification_model.predict(face_image)
        
        if (member_predict.argmax() == member_index) and (member_predict[0][member_index] > confidence):
            captured_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            img_name = f"{IZ_ONE_index_map[member_index]}_{captured_time}.png"
            cv2.imwrite(os.path.join(save_dir, img_name), image)
            print(f"predict : {member_predict.argmax()}, input : {member_index}")
            print(f"predict confidence : {member_predict[0][member_index]}, input : {confidence}")
#             plt.imshow(face_image[0])
#             plt.show()
            return

def youtube_video_parsing(url, member_index, save_dir, member_classification_model):
    global loading_chk

    video = pafy.new(url)
    print('title = ', video.title)
    print('video.rating = ', video.rating)
    print('video.duration = ', video.duration)
        
    best = video.getbestvideo(preftype='webm')
    print('best.resolution', best.resolution)
    
    cap=cv2.VideoCapture(best.url)

    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('frame_size =', frame_size)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out1 = cv2.VideoWriter('./data/record0.mp4',fourcc, 20.0, frame_size)
    out2 = cv2.VideoWriter('./data/record1.mp4',fourcc, 20.0, frame_size,isColor=False)
    
    prev_time = 0
    
    FPS = 10

    while True:
        retval, frame = cap.read()
        if not retval:
            break   

        current_time = time.time() - prev_time
        
        if (retval == True) and (current_time > 1./FPS):
            prev_time = time.time()
            
            face_detect_and_member_classification(frame, member_index, save_dir, member_classification_model)
       
    cap.release()
    out1.release()
    out2.release()
    
    loading_chk = False
    
def intranet_video_parsing(path, member_index, save_dir, member_classification_model):
    global loading_chk
    
    if os.path.isfile(path):
        cap = cv2.VideoCapture(path)
    else:
        raise f"{path}에 영상이 존재하지 않습니다."

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    frameRate = 33
    
    prev_time = 0
    FPS = 10

    while True:
        retval, frame = cap.read()
        if not(retval):
            break
        
        current_time = time.time() - prev_time
        
        if (retval == True) and (current_time > 1./FPS):
            prev_time = time.time()
            
            face_detect_and_member_classification(frame, member_index, save_dir, member_classification_model)
            
    if cap.isOpened():
        cap.release()
        
    loading_chk = False
        
def local_video_parsing(path, member_index, save_dir, member_classification_model):
    global loading_chk
    
    filePath = path
    print(filePath)

    if os.path.isfile(filePath):
        cap = cv2.VideoCapture(filePath)
    else:
        raise f"{filePath}에 영상이 존재하지 않습니다."

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    frameRate = 33
    
    prev_time = 0
    FPS = 10

    while True:
        retval, frame = cap.read()
        if not(retval):
            break
        
        current_time = time.time() - prev_time
        
        if (retval == True) and (current_time > 1./FPS):
            prev_time = time.time()
            
            face_detect_and_member_classification(frame, member_index, save_dir, member_classification_model)
            
    if cap.isOpened():
        cap.release()
        
    loading_chk = False

window.mainloop()


# https://youtu.be/XHr7vaHJvq4

# # refactoring

# In[11]:


# class IZ_ONE_Finder(tk.Tk):
#     def __init__(self):
        
#     def initUI(self):
#         window_width = 1200
#         window_height = 895
        
#         screen_width = self.winfo_screenwidth()
#         screen_height = self.winfo_screenheight()
        
#         x_cordinate = int((screen_width/2) - (window_width/2))
#         y_cordinate = int((screen_height/2) - (window_height/2))

