from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
# import pytesseract
import easyocr
import cv2
import imutils
import math

'''# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
#
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
#
# You can return the answer in any order.
def twoSum(array, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """

    # a = len(array)
    insert_list = list()
    for i in range(len(array)):
        j = i + 1
        ans = array[i] + array[j]
        if ans == target:
            insert_list.append(i)
            insert_list.append(j)
    return insert_list

array = [2, 4, 1, 8, 5, 1, 3, 4]
target = 6
twoSum(array, target)'''


# def twoSum(nums, target):# -> list[int]:
#     for i in range(len(nums)):
#         for j in range(i + 1, len(nums)):
#             if nums[j] == target - nums[i]: # complicated logic
#                 return [i, j]
#
# array = [2, 4, 1, 8, 5, 1, 3, 4]
# target = 6
# twoSum(nums=array, target=target)

'''Given an integer x, return true if x is palindrome integer.

An integer is a palindrome when it reads the same backward as forward.

For example, 121 is a palindrome while 123 is not.'''

# x = 13431
# y = 12341
# # rev = int(str(x)[::-1])
# # rev = int(rev)
# # print(x, str(x)[::-1])
# if int(str(x)[::-1]) == x:
#     print(f"{x} is palindrome")
# else:
#     print(f"{x} is not palindrome")


'''#ocr for ANPR
cropped_image = cv2.imread('image2.jpg')
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

imgplot = plt.imshow(cropped_image)
plt.show()

imgplot = plt.imshow(gray)
plt.show()

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

imgplot = plt.imshow(edged)
plt.show()

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)
'''
# li = list([1, 2,3,4])
# print(li[1])

'''
Roman number to simple numbers
Dict = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
print("enter roman numbers")
roman = input(str()).upper()
spliting = [char for char in roman]
if len(roman) == 1:
    print(Dict[roman])
else:
    char1st = 0
    for i in range(len(spliting)-1):
        if Dict[spliting[i]] < Dict[spliting[i+1]]:
            char1st = char1st + (Dict[spliting[i + 1]] - Dict[spliting[i]])
        else:
            char1st = char1st + (Dict[spliting[i + 1]] + Dict[spliting[i]])
    print(char1st)'''
    # char1st = Dict[spliting[0]]
    # for i in range(1, len(spliting)):
    #     char1st = char1st + Dict[spliting[i]]
    # print(char1st)

''' merge two lists and sort them
list1 = list([3, 6,8, 19, 21])
list2 = list([2,6,6,7,8,20])
list3 = (list1 + list2)
print(sorted(list3))'''

''''
remove duplicate from sorted list
list1 = list([2, 3, 6, 6, 7, 8, 8, 19, 20, 21])
list2 = list()
for i in range(len(list1)):
    list2.append("_")
# # print(list2)
i = 0
for character in (list1):
    if character in list2:
        None
    else:
        list2[i] = character
        i+=1
print(list2)'''

'''# median of two sorted arrays
num1 = [1, 3, 5, 4]
num2 = [2, 6]
mergelist = list(set(num1 + num2))
n = len(mergelist)
if n % 2 == 0:
    even_index1 = int(n/2)
    even_index2 = even_index1-1
    even = (mergelist[even_index1] + mergelist[even_index2]) / 2
    print(even)
else:
    odd_index = math.floor((n + 1) / 2)
    odd = mergelist[odd_index]
    print(odd_index)
'''

# a = "pwwkew"
# b = str(set(a))
# print(b)

# cropped_image = cv2.imread('image2.jpg')
# gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#
# imgplot = plt.imshow(cropped_image)
# plt.show()

cropped_image = cv2.imread('mask1.jpg')
# imgplot = plt.imshow(cropped_image)
# plt.show()
# cropped_image = np.uint8(cropped_image)
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

imgplot = plt.imshow(cropped_image)
plt.show()

imgplot = plt.imshow(gray)
plt.show()

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

imgplot = plt.imshow(edged)
plt.show()

keypoints, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(gray.shape, np.uint8)
mask.fill(255)
for keys in range(len(keypoints)):
    cv2.drawContours(mask, keypoints, keys, (0,0,0), 3)
    cv2.waitKey(0)
    cv2.imshow("original", mask)
    # imgplt = plt.imshow(mask)
    # plt.show()
# contours = imutils.grab_contours(keypoints)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# location = None
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 10, True)
#     if len(approx) == 4:
#         location = approx
#         break

# mask = np.zeros(gray.shape, np.uint8)
# new_image = cv2.drawContours(mask, [location], 0, 255, -1)
# new_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
#
# (x, y) = np.where(mask == 255)
# (x1, y1) = (np.min(x), np.min(y))
# (x2, y2) = (np.max(x), np.max(y))
# cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# reader = easyocr.Reader(['en'])
# result = reader.readtext(cropped_image)
# print(result)