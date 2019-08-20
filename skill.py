import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./test_pictures/test4.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#bottom_region = bars_region(img)
money_region = img[1045:1070, 1200:1280]
money_region_gray = cv2.cvtColor(money_region, cv2.COLOR_BGR2GRAY)
_, money_region_binary = cv2.threshold(money_region_gray, 200, 255, cv2.THRESH_BINARY)
#cv2.imwrite('./raw.png',cv2.resize(bottom_region_binary, (480, 180)))
#######got the money########
money = money_region_binary

#######get the circle on the right#######
#######right panel region: X (1120,1327), Y(930, 1080)

circles = cv2.HoughCircles(img_gray[930:1080, 1120:1327],cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=15,maxRadius=20)
if (circles is not None):
    print(circles)
else:
    print('no circles found')

########SKILLS STATE########
class skill:
    def __init__(self, loc, name, upgradability):
        self.loc = loc
        self.name = name
        self.img = img_hsv_thresh
        self.upgradability = upgradability
    def get_state(self):
        brightness = cv2.mean(self.img[self.loc[0]:self.loc[1],self.loc[2]:self.loc[3]])
        #if the skill level is not 0, which means not developed
        #Determine whether it is cooling or ready
        if (self.get_skill_level() != 0):
            return 'Cooling' if (brightness[0] < 20) else 'Ready'
        else:
            #otherwise, it is not developed
            return 'Not Developed'
    def get_img(self):
        return img[self.loc[0]:self.loc[1],self.loc[2]:self.loc[3]]
    def get_skill_level(self):
        if (self.upgradability == True):
            #getting the brightness/luminus of the skill dot area
            luminus = cv2.sumElems(self.img[1012:1019,self.loc[2]:self.loc[3]])
            #One skill dot worths about 9754 luminus
            skill_level = round(luminus[0]/9754)
            return skill_level
        else:
            return None

low_limit = np.array([0, 0, 128])
high_limit = np.array([180, 255, 255])
img_hsv_thresh = cv2.inRange(img_hsv, low_limit, high_limit)
##
skill_0 = skill([950, 990, 680, 722], '0', False)
skill_Q = skill([950, 1005, 731, 787], 'Q', True)
skill_W = skill([950, 1005, 798, 853],'W', True)
skill_E = skill([950, 1005, 861, 921],'E', True)
skill_R = skill([950, 1005, 930, 990],'R', True)
skill_D = skill([950, 990, 1005, 1048],'D', False)
skill_F = skill([950, 990, 1055, 1100],'F', False)
skills = [skill_0, skill_Q, skill_W, skill_E, skill_R, skill_D, skill_F]
#CHECK THE AVERAGE BRIGHTNESS
print("_________Skill State_________")
for skill in skills :
    print(skill.get_state())
print("_____________________________")
#######
plt.figure()
for i in range(7) :
    plt.subplot(1,7,i+1)
    plt.imshow(skills[i].get_img())
    plt.title(skills[i].name + ': ' + skills[i].get_state())
    plt.xlabel(str(skills[i].get_skill_level()))


plt.figure()
plt.imshow(img)
plt.show()
'''
plt.figure()
plt.imshow(img_hsv_thresh)

plt.figure()
plt.imshow(img_gray[930:1080, 1120:1327])
plt.show()
'''
