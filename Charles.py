#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image as pim

SKY = 0
FLOWER = 1
STEM = 2
JUNCTION = 3

black = (0,0,0)
blue = (200,200,255)
brown = (100,50,0)
green = (50,150,0)
orange = (255,150,0)
yellow = (255,255,0)


# In[2]:


BLOCK_SIZE = 3

def draw_flower(img, x, y):
    img.putpixel((x+1,y+0), yellow)
    img.putpixel((x+1,y+2), yellow)
    img.putpixel((x+0,y+1), yellow)
    img.putpixel((x+2,y+1), yellow)
    img.putpixel((x+1,y+1), orange)

def draw_sky(img, x, y):
    pass

def draw_stem(img, x, y):
    img.putpixel((x+1,y+0), green)
    img.putpixel((x+1,y+1), green)
    img.putpixel((x+1,y+2), green)

def draw_junction_left(img, x, y):
    img.putpixel((x-1, (y+0) % img.width), green)
    img.putpixel((x+0, (y+1) % img.width), green)
    img.putpixel((x+1, (y+2) % img.width), green)

def draw_junction_right(img, x, y):
    img.putpixel((x+3, (y+0) % img.width), green)
    img.putpixel((x+2, (y+1) % img.width), green)
    img.putpixel((x+1, (y+2) % img.width), green)
    

drawers = {SKY: draw_sky, FLOWER: draw_flower, STEM: draw_stem}


# In[3]:


def visualise(inv_mat):
    mat = []
    i = len(inv_mat) - 1
    while i >= 0:
        mat.append(inv_mat[i])
        i -= 1
    
    width = len(mat[0])
    height = len(mat)
    img = pim.new('RGB', (width * BLOCK_SIZE, (height + 1) * BLOCK_SIZE))
    img.paste(blue, (0, 0, img.width, img.height))

    # draw not-junctions
    for x in range(width):
        for y in range(height):
            if mat[y][x] == JUNCTION:
                continue
            drawers[mat[y][x]](img, x*BLOCK_SIZE, y*BLOCK_SIZE)
    # draw junctions
    for x in range(width):
        for y in range(height):
            if y > 0 and mat[y][x] == JUNCTION:
                left = right = False
                if mat[y-1][(x-1) % width] >= FLOWER:
                    draw_junction_left(img, x*BLOCK_SIZE, y*BLOCK_SIZE)
                if mat[y-1][(x+1) % width] >= FLOWER:
                    draw_junction_right(img, x*BLOCK_SIZE, y*BLOCK_SIZE)
    # draw dirt
    img.paste(brown, (0, height*BLOCK_SIZE, img.width, img.height))

    scale = 10
    scaled = pim.new(img.mode, (img.width*scale, img.height*scale))
    for x in range(img.width):
        for y in range(img.height):
            scaled.paste(img.getpixel((x, y)), (x*scale, y*scale, (x+1)*scale, (y+1)*scale))
    return scaled


# In[ ]:


if __name__ == '__main__':
    test_mat = [[0, 2, 0, 2, 0, 0, 0, 0, 2, 0],
             [0, 3, 0, 1, 0, 0, 0, 0, 3, 0],
             [0, 0, 2, 0, 0, 0, 0, 3, 0, 1],
             [0, 0, 2, 0, 0, 0, 2, 0, 2, 0],
             [0, 0, 2, 0, 0, 0, 3, 0, 3, 0],
             [0, 0, 2, 0, 0, 0, 0, 3, 0, 2],
             [0, 0, 3, 0, 0, 0, 0, 0, 1, 2],
             [0, 1, 0, 3, 0, 0, 0, 0, 0, 1],
             [0, 0, 2, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    visualise(test_mat)

