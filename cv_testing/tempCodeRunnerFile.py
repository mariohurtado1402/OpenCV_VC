for i in range(len(corners)):
#     for j in range(i + 1, len(corners)):
#         corner1 = tuple(map(int, corners[i].ravel()))
#         corner2 = tuple(map(int, corners[j].ravel()))
#         color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
#         cv2.line(img, corner1, corner2, color, 1)