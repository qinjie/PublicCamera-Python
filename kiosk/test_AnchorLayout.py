from kivy.app import App
from kivy.uix.image import Image
from kivy.config import Config
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout

class MainScreen(GridLayout):

    def __init__(self, **kwargs):

        super(MainScreen, self).__init__(**kwargs)

        self.spacing = [5, 10]
        self.float_list = []
        self.anchor_list = []
        self.image_list = []
        self.relative_list = []
        self.rows = 2

        image = Image(source='./tmp_pics/1.jpg', size_hint=(None, None))
        print 'image:', image.texture.size


        for i in range(4):
            if i % 2 == 0:
                self.anchor_list.append(AnchorLayout(anchor_x='right'))
            else:
                self.anchor_list.append(AnchorLayout(anchor_x='left'))

            self.image_list.append(Image(source='./tmp_pics/1.jpg', size_hint=(None, None)))

            self.anchor_list[i].add_widget(self.image_list[i])

            self.relative_list.append(RelativeLayout())

            self.relative_list[i].add_widget(self.anchor_list[i])

            self.add_widget(self.relative_list[i])


class MainDisplay(App):

    def __init__(self):
        super(MainDisplay, self).__init__()

    def build(self):

        self.mainScreen = MainScreen()

        return self.mainScreen

if __name__ == '__main__':

    Config.set('graphics', 'width', '1600')
    Config.set('graphics', 'height', '900')

    app = MainDisplay()
    app.run()