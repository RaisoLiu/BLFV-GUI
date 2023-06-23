import tkinter as tk
from tkinter import Frame, Entry, Label
from tkinter import filedialog, colorchooser
from tkinter import Scrollbar, Canvas
from video_processing import load_video
from feature_extraction import FeatureFilter
import os
import cv2
import numpy as np
from feature_extraction import PatchFeatureGenerator   
from video_processing import VideoManager
from PIL import Image, ImageTk
from tkinter import ttk
from matplotlib import pyplot as plt
kernel_frame_size = 4


# transform

import torchvision.transforms as tt
resolution = 518
patch_len = resolution // 14

img2tensor = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution) ),
    tt.Normalize(mean=0.5, std=0.2), # range [0.0,1.0] -> [-2.5, 2.5]

])

tensor2img = tt.Compose([
    tt.Normalize(mean=-2.5, std=5), # range [-2.5, 2.5] -> [0.0,1.0]
    tt.ToPILImage()
])

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

class ApplicationWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Bodypart Latent From Video")
        self.frames = []
        self.entries = {}
         # 創建工具列
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.modelmenu = tk.Menu(self.menubar, tearoff=0)
        self.toolmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.menubar.add_cascade(label="Model Load", menu=self.modelmenu)
        self.menubar.add_cascade(label="Tool", menu=self.toolmenu)
        self.filemenu.add_command(label="Select File", command=self.select_file)
        self.toolmenu.add_cascade(label="Color Look up", command=self.check_color)
        self.modelmenu.add_command(label="dinov2_vits14", command=self.select_dinov2_vits14)
        self.modelmenu.add_command(label="dinov2_vitb14", command=self.select_dinov2_vitb14)
        
        self.config(menu=self.menubar)
        
        for i in range(7):
            self.columnconfigure(i, weight=1)

        for i in range(6):
            self.rowconfigure(i, weight=1)

        self.video_player = tk.Label(self, text='BLVF', bg='white')
        self.video_player.grid(row=0,column=0,rowspan=6, columnspan=6, sticky='nsew')

        self.video_controller = ttk.Scale(self, command=self.set_frame)
        self.video_controller.grid(row=6,column=0,rowspan=1, columnspan=6, sticky='nsew')
        self.video_controller.configure(state='disabled') 




        self.process_block = tk.Frame(self, bg='orange')
        self.process_block.grid(row=0,column=6,rowspan=4, columnspan=2, sticky='nsew')  

        self.process_toolbox = tk.Frame(self, bg='red')
        self.process_toolbox.grid(row=4,column=6,rowspan=1, columnspan=2, sticky='nsew')   
        self.process_toolbox.rowconfigure(0, weight=1)
        for i in range(3):
            self.process_toolbox.columnconfigure(i, weight=1)

        self.addButton1 = tk.Button(self.process_toolbox, text="PCA-1D", command=self.add1dSubbox)
        self.addButton1.grid(row=0, column=0, sticky='nsew')


        self.addButton2 = tk.Button(self.process_toolbox, text="PCA-3D", command=self.add3dSubbox)
        self.addButton2.grid(row=0, column=1, sticky='nsew')


        self.deleteButton = tk.Button(self.process_toolbox, text="POP", command=self.deleteLastSubbox)
        self.deleteButton.grid(row=0, column=2, sticky='nsew')




        self.export_menu = tk.Label(self, text='export_menu', bg='orange')
        self.export_menu.grid(row=5,column=6,rowspan=3, columnspan=2, sticky='nsew')   

        
        self.status_bar = tk.Label(self, text='Welcome to Bodypart Latent from video', borderwidth=2)
        self.status_bar.grid(row=7,column=0,rowspan=1, columnspan=6, sticky='nsew')


        self.filename = None
        self.frame_style = 0
        self.is_init_kernel_frames = None
        self.log = self.update_status_bar 
        self.pfg = None
        self.bind('<Left>', self.move_slider_left)
        self.bind('<Right>', self.move_slider_right)

        # 在process_block中建立滾動條和畫布
        self.canvas = Canvas(self.process_block)
        self.scrollbar = Scrollbar(self.process_block, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        # 將滾動條配置到畫布並設定滾動區域
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 打包畫布和滾動條
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


        self.bind('<space>', lambda event: self.toggle_frame_style())



    def select_file(self):
        self.filename = filedialog.askopenfilename()
        print(self.filename)
        self.export = self.filename.split('.')[0]
        os.makedirs(self.export, exist_ok=True)
        self.vm = VideoManager(self.filename, self.export, kernel_frame_size)
        self.current_frame_index = 0
        self.update_video_player()
        self.video_controller.configure(state='active')  # 'normal'
        self.ff = FeatureFilter()

        while self.frames:
            frame = self.frames.pop()
            del self.entries[frame]
            frame.destroy()

        self.last_dim_pca = None

    def resetFeatureFilter(self):
        while self.frames:
            frame = self.frames.pop()
            del self.entries[frame]
            frame.destroy()
        self.ff = FeatureFilter()
        self.last_dim_pca = None
        self.frame_style = 0


    def select_dinov2_vits14(self):
        self.log("Loading dinov2_vits14 (small)")
        self.pfg = PatchFeatureGenerator('dinov2_vits14')
        self.patch_size = self.pfg.model.patch_size
        self.num_features = self.pfg.model.num_features
        self.resetFeatureFilter()

    def select_dinov2_vitb14(self):
        self.log("Loading dinov2_vitb14 (base)")
        self.pfg = PatchFeatureGenerator('dinov2_vitb14')
        self.patch_size = self.pfg.model.patch_size
        self.num_features = self.pfg.model.num_features
        self.resetFeatureFilter()

    def update_video_player(self):
        self.current_frame = self.vm.getframe(self.current_frame_index)
        if self.frame_style == 0:
            self.current_frame = cv2.cvtColor(np.array(self.current_frame), cv2.COLOR_RGB2BGR)
            self.current_frame = Image.fromarray(self.current_frame)
            self.current_frame = ImageTk.PhotoImage(self.current_frame)
            self.video_player.config(image=self.current_frame)
        else:
            tensor = img2tensor(self.current_frame)
            patch_feature = self.pfg.single_run(tensor)
            X = patch_feature.reshape((-1, self.num_features))
            mm, zz = self.ff.getFeature(X)
            frames_feat = np.zeros((self.n_token_per_frame, self.last_dim_pca))
            try:
                frames_feat[mm] = zz
            except:
                pass
            if self.frame_style == 1:
                plt.imshow(self.current_frame, extent=(0, resolution, resolution,0))
                plt.imshow(frames_feat.reshape((patch_len, patch_len, self.last_dim_pca)), extent=(0, resolution, resolution,0), alpha=0.3, cmap="inferno")
      
            if self.frame_style == 2:
                plt.imshow(frames_feat.reshape((patch_len, patch_len, self.last_dim_pca)), extent=(0, resolution, resolution,0), alpha=1, cmap="inferno")
            plt.axis('off')

            fig = plt.gcf()
            img = fig2img(fig)
            plt.close()
            self.current_frame = ImageTk.PhotoImage(img)
            self.video_player.config(image=self.current_frame)
            self.frames_feat = frames_feat

    def update_status_bar(self, msg):
        self.status_bar.config(text=msg)

    def set_frame(self, value):
        # print('v', value)
        if self.filename:
            try:
                self.current_frame_index = int(int(self.vm.n_frame) * float(value))
                self.update_video_player()
                # print('self.current_frame', self.current_frame_index)
            except:
                pass
        self.log("current_frame_index: " + str(self.current_frame_index))

    def move_slider_left(self, event):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.update_video_player()
        self.log("current_frame_index: " + str(self.current_frame_index))
        
        # self.video_controller.set(current_value - 1)


    def move_slider_right(self, event):
        if self.current_frame_index < self.vm.n_frame - 1:
            self.current_frame_index += 1
            self.update_video_player()
        self.log("current_frame_index: " + str(self.current_frame_index))

    def add3dSubbox(self):
        if not self.is_init_kernel_frames:
            self.init_kernel_frames()


        frame = Frame(self.scrollable_frame, bg="red", bd=2, relief="solid")
        frame.pack(fill='x', padx=5, pady=5, in_=self.scrollable_frame)
        
        self.frames.append(frame)
        self.entries[frame] = []
        
        labels = ['R', 'G', 'B']
        for j in range(3):
            Label(frame, text=labels[j]).grid(row=0, column=j)

        for i in range(2):
            for j in range(3):
                entry = Entry(frame, width=5)
                entry.grid(row=i+1, column=j, padx=2, pady=2)
                entry.insert(0, '1.00' if i == 0 else '0.00')
                entry.bind("<Up>", self.increment_value)
                entry.bind("<Down>", self.decrement_value)
                self.entries[frame].append(entry)
        
        self.ff.addLayer(self.X, 3)
        self.last_dim_pca = 3
        self.update_video_player()

    def add1dSubbox(self):
        if not self.is_init_kernel_frames:
            self.init_kernel_frames()

        frame = Frame(self.scrollable_frame, bg="blue", bd=2, relief="solid")
        frame.pack(fill='x', padx=5, pady=5, in_=self.scrollable_frame)
        
        self.frames.append(frame)
        self.entries[frame] = []
        
        Label(frame, text='TH').grid(row=0, column=0)

        for i in range(2):
            entry = Entry(frame, width=5)
            entry.grid(row=i+1, column=0, padx=2, pady=2)
            entry.insert(0, '1.00' if i == 0 else '0.00')
            entry.bind("<Up>", self.increment_value)
            entry.bind("<Down>", self.decrement_value)
            entry.bind("<Return>", self.keyin_value) 
            self.entries[frame].append(entry)
        
        self.ff.addLayer(self.X, 1)
        self.last_dim_pca = 1
        self.update_video_player()


    def deleteLastSubbox(self):
        if self.frames:
            frame = self.frames.pop()
            del self.entries[frame]
            frame.destroy()
            self.ff.rmLayer()
        
        if len(self.ff.blk) > 0:
            self.last_dim_pca = self.ff.blk[-1].n_pca
        else:
            self.last_dim_pca = None

        self.update_video_player()

    def keyin_value(self, event):
        self.adjust_value(event.widget, .0)
    
    def increment_value(self, event):
        self.adjust_value(event.widget, 0.05)

    def decrement_value(self, event):
        self.adjust_value(event.widget, -0.05)

    def adjust_value(self, entry, delta):
        value = float(entry.get())
        value = min(max(value + delta, 0.0), 1.0)
        value = format(value, '.2f')
        entry.delete(0, 'end')
        entry.insert(0, value)

        lastid = len(self.ff.blk) - 1
        v = self.get_values(lastid)
        th = [(float(v[i+self.last_dim_pca]), float(v[i])) for i in range(self.last_dim_pca)]
        print(th, type(v[0]))
        self.ff.setLayerThreshold(lastid, th)
        self.update_video_player()

    def get_values(self, frame_index):
        frame = self.frames[frame_index]
        return [entry.get() for entry in self.entries[frame]]
    
    def init_kernel_frames(self):
        self.log("Init PCA")
        self.is_init_kernel_frames = True
        if not self.pfg:
            self.select_dinov2_vits14()

        kernel_frame = self.vm.getkernelframe()
        kernel_tensor = [img2tensor(it) for it in kernel_frame]
        patch_feature = self.pfg.batch_run(kernel_tensor)
        X = np.array(patch_feature)
        n_kernel, self.n_token_per_frame, n_feat = X.shape
        print(n_kernel, self.n_token_per_frame, n_feat)
        X = X.reshape((-1, n_feat))
        self.log(f"{X.shape}")
        self.X = X

    def toggle_frame_style(self):
        self.frame_style = (self.frame_style + 1) % 3
        self.log(f"frame_style is now: {self.frame_style}")  # 可以移除這行，只是用來驗證 frame_style 是否已經切換
        self.update_video_player()

    def check_color(self):
        color, _ = colorchooser.askcolor()
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        self.log(f"R={r:.2f}, G={g:.2f}, B={b:.2f}")


