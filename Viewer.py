from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from Picker import createShader
import numpy as np
import math
import time


# 컴그
# 수조 불러오기
# 초기 빛 경로 설정 -200 부터 200까지 
# 스틱 점 100개 연결된 것으로 설정 -> 메쉬로 하기 너무 힘들고 렌더링도 힘들 것 같았음(답이 안보였음..)
# 오른쪽클릭으로 수조 생성, 3d 이외의 화면에서 위치 변경
# fov랑 스케일은 고정된 값으로 진행(fov = 90, 스케일 = 1)
# z로 스틱의 시작점, x로 스틱의 끝 점 지정 가능, 화면 클릭하고 z or x 누르고 콘솔에서 x y z 입력
# 세팅 완료 후 화면에서 q 누르고 3d 화면 한 번 클릭(딱 한 번! 무조건 한번!)
vnsujo = []
vsujo = []
fsujo = []
tnsujo = []

switch = False

rays = np.array([[[i / 100, j / 100, -2, 0] for j in range(-200, 201)] for i in range(-200, 201)])

startpoint = np.array([-0.5, 0.5, 0.5])
endpoint = np.array([0, 0, 0])
stickpoints = np.array([(startpoint * (100 - i) / 100 + endpoint * i / 100) for i in range(101)])
drawingdestpoints = []

f = open("sujo.obj", 'r', encoding='utf-8')
while True:
    line = f.readline()
    if not line:
        break
    if line[0:2] == "vn":
        vnsujo.append(list(map(float, line.split()[1:])))
    elif line[0] == "v":
        vsujo.append(list(map(float, line.split()[1:])))
        vsujo[-1][0] /= 25
        vsujo[-1][1] /= 25
        vsujo[-1][2] /= 25
    if line[0] == "f":
        t = line.split()[1:]
        fsujo.append([int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1])
        a, b, c = np.array(vsujo[fsujo[-1][0]]), np.array(vsujo[fsujo[-1][1]]), np.array(vsujo[fsujo[-1][2]])
        tn = np.cross(np.array(b) - np.array(a), np.array(c) - np.array(a))
        tnsujo.append(tn / np.linalg.norm(tn))
f.close()

oneD_vsujo = np.array(vsujo).flatten()
oneD_fsujo = np.array(fsujo).flatten()
oneD_tnsujo = np.array(tnsujo).flatten()

class Object:
    cnt = 0

    def __init__(self):
        # Do NOT modify: Object's ID is automatically increased
        self.id = Object.cnt
        Object.cnt += 1
        # self.mat needs to be updated by every transformation
        self.mat = np.eye(4)

    def draw(self):
        raise NotImplementedError

    #Rotation about each plane. Store and restore the original coordinates after rotation using matrix multiplication.
    #For example, because below rotation's origin is z-axis, we store x and y then do some calculation.
    #[[cos(a), sin(a), 0, 0], [-sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] is rotation matrix about xy plane.
    def xyRotation(self, degree):
        original_x = self.mat[0, 3]
        original_y = self.mat[1, 3]
        self.translation(-original_x, -original_y, 0)
        t = np.eye(4)
        t[0, 0] = math.cos(math.radians(degree))
        t[0, 1] = math.sin(math.radians(degree))
        t[1, 0] = -math.sin(math.radians(degree))
        t[1, 1] = math.cos(math.radians(degree))
        self.mat =  t @ self.mat
        self.translation(original_x, original_y, 0)

    #[[1, 0, 0, 0], [0, cos(a), sin(a), 0], [0, -sin(angle), cos(a), 0], [0, 0, 0, 1]] is rotation matrix about yz plane.
    def yzRotation(self, degree):
        original_y = self.mat[1, 3]
        original_z = self.mat[2, 3]
        self.translation(0, -original_y, -original_z)
        t = np.eye(4)
        t[1, 1] = math.cos(math.radians(degree))
        t[1, 2] = math.sin(math.radians(degree))
        t[2, 1] = -math.sin(math.radians(degree))
        t[2, 2] = math.cos(math.radians(degree))
        self.mat =  t @ self.mat
        self.translation(0, original_y, original_z)

    #[[cos(a), 0, -sin(a), 0], [0, 1, 0, 0], [sin(a), 0, cos(a), 0], [0, 0, cos(a), 1]] is rotation matrix about zx plane.
    def zxRotation(self, degree):
        original_x = self.mat[0, 3]
        original_z = self.mat[2, 3]
        self.translation(-original_x, 0, -original_z)
        t = np.eye(4)
        t[2, 2] = math.cos(math.radians(degree))
        t[2, 0] = math.sin(math.radians(degree))
        t[0, 2] = -math.sin(math.radians(degree))
        t[0, 0] = math.cos(math.radians(degree))
        self.mat =  t @ self.mat
        self.translation(original_x, 0, original_z)

    #Scaling. Because scale must be done by using objects' center as origin, we store x ,y  and z then do some calculation.
    #[[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]] is scaling matrix.
    def scaling(self, sx, sy, sz):
        original_x = self.mat[0, 3]
        original_y = self.mat[1, 3]
        original_z = self.mat[2, 3]
        self.translation(-original_x, -original_y, -original_z)
        t = np.eye(4)
        t[0, 0] = sx
        t[1, 1] = sy
        t[2, 2] = sz
        self.mat =  t @ self.mat
        self.translation(original_x, original_y, original_z)

    #Translation.
    #[[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]] is translation matrix.
    def translation(self, dx, dy, dz):
        t = np.eye(4)
        t[0, 3] = dx
        t[1, 3] = dy
        t[2, 3] = dz
        self.mat =  t @ self.mat

# 수조 오브젝트
# checkSujo는 해당 경로로 이동하고 있는 빛이 수조와 맞닿는지 확인
# 확인 되면 굴절을 위해 닿는 지점, index, normal을 반환
class Sujo(Object):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.mat.T)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, oneD_vsujo)
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, oneD_tnsujo)
        glDrawElements(GL_TRIANGLES, len(oneD_fsujo), GL_UNSIGNED_INT, oneD_fsujo)
        glPopMatrix()

    def checkSujo(self, start, dest):
        t = self.mat
        transv = np.matmul(t, np.append(np.array(vsujo).T, np.array([1 for _ in range(len(vsujo))])).reshape(4, len(vsujo)))
        mx, Mx, my, My, mz, Mz = min(transv[0]), max(transv[0]), min(transv[1]), max(transv[1]), min(transv[2]), max(transv[2])
        x, y, z = start[0], start[1], start[2]
        nx, ny, nz = dest[0], dest[1], dest[2]
        v = dest - start
        coor = [1234]
        normal = []
        if x < mx < nx or x > mx > nx:
            d = abs((mx - x) / v[0])
            if my < y + v[1] * d < My and mz < z + v[2] * d < Mz:
                coor.append(np.array([mx, y + v[1] * d, z + v[2] * d]))
                normal.append(np.array([-1, 0, 0]))
        if x > Mx > nx or x < Mx < nx:
            d = abs((Mx - x) / v[0])
            if my < y + v[1] * d < My and mz < z + v[2] * d < Mz:
                coor.append(np.array([Mx, y + v[1] * d, z + v[2] * d]))
                normal.append(np.array([1, 0, 0]))
        if y < my < ny or y > my > ny:
            d = abs((my - y) / v[1])
            if mx < x + v[0] * d < Mx and mz < z + v[2] * d < Mz:
                coor.append(np.array([x + v[0] * d, my, z + v[2] * d]))
                normal.append(np.array([0, -1, 0]))
        if y > My > ny or y < My < ny:
            d = abs((My - y) / v[1])
            if mx < x + v[0] * d < Mx and mz < z + v[2] * d < Mz:
                coor.append(np.array([x + v[0] * d, My, z + v[2] * d]))
                normal.append(np.array([0, 1, 0]))
        if z < mz < nz or z > mz> nz:
            d = abs((mz - z) / v[2])
            if mx < x + v[0] * d < Mx and my < y + v[1] * d < My:
                coor.append(np.array([x + v[0] * d, y + v[1] * d, mz]))
                normal.append(np.array([0, 0, -1]))
        if z > Mz > nz or z < Mz < nz:
            d = abs((Mz - z) / v[2])
            if mx < x + v[0] * d < Mx and my < y + v[1] * d < My:
                coor.append(np.array([x + v[0] * d, y + v[1] * d, Mz]))
                normal.append(np.array([0, 0, 1]))
        if len(coor) > 1:
            return [coor, self.index, normal]
        else:
            return False
        


class SubWindow:
    """
    SubWindow Class.\n
    Used to display objects in the obj_list, with different camera configuration.
    """

    #My code need selected objects set, mouse moving state variable.
    #current positions for constant transformation.
    #fov and scale variables.
    windows = []
    obj_list = []
    selected = set()
    moving = ""
    currentX = 0
    currentY = 0
    fov = 90
    scale = 1
    
    def __init__(self, win, x, y, width, height):
        # identifier for the subwindow
        self.id = glutCreateSubWindow(win, x, y, width, height)
        # projection matrix
        self.projectionMat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # view matrix, you do not need to modify the matrix for now
        self.viewMat = np.eye(4)
        # shader program used to pick objects, and its associated value. DO NOT MODIFY.
        self.pickingShader, self.pickingColor = createShader()
        self.width = width
        self.height = height
        if self.id == 5:
            self.projection()

    def display(self):
        """
        Display callback function for the subwindow.
        """
        glutSetWindow(self.id)

        self.drawScene()

        glutSwapBuffers()

    def drawScene(self):
        """
        Draws scene with objects to the subwindow.
        """
        glutSetWindow(self.id)

        glClearColor(0.8, 0.8, 0.8, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)

        

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projectionMat.T)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.viewMat.T)

        self.drawAxes()

        glBegin(GL_LINES)
        glColor3f(0, 0, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glEnd()

        global drawingdestpoints, rays, stickpoints, switch
        if self.id == 5 and switch:
            # switch가 되어있으면 굴절된 화면을 표시
            # drawingdestpoints에 굴절된 점들 위치를 저장하고 그리기
            # 카메라는 빛 추적을 시작할 위치
            # x는 빛의 원래 경로를 카메라 뷰에 따라 회전시켜 보는 위치에 맞게 설정
            # dests는 x와 camera 위치를 더해 목적지 위치를 저장
            drawingdestpoints = []
            t = SubWindow.windows[3].viewMat.T
            camera = t[:3, 3] + t[:3, 2]
            x = np.matmul(t, rays.reshape(160801, 4).T)[:3].T
            dests = x + camera
            
            # 모든 빛에 대한 계산 시작
            for i in range(len(x)):
                #진행도 확인을 위해
                if i % 401 == 0:
                    print(i // 401)
                # 해당 빛에 대한 목적지, 시작지점, 마지막 표시를 위한 원래 목적지(굴절하면 목적지가 바뀌므로)
                # 공기의 index를 현재 index로
                dest = dests[i]
                start = camera
                original_dest = dest
                current_index = 1
                while True:
                    #계산 시작, points는 가장 가까운 위치의 후보지들(수조의 면 혹은 스틱의 점)
                    points = []
                    dist = []
                    # 모든 수조에 대해
                    for obj in SubWindow.obj_list:
                        #수조와 충돌하는지 확인하고 그렇다면 points에 저장
                        t = obj.checkSujo(start, dest)
                        if t:
                            x = t[1:]
                            x.insert(0, t[0][1])
                            x[2] = t[2][0]
                            points.append(x)
                            if len(t[0]) == 3:
                                x = t[1:]
                                x.insert(0, t[0][2])
                                x[2] = t[2][1]
                                points.append(x)
                    # y는 점과 시작점의 경로와 원래 경로가 겹치는지 확인(경로 중간에 점이 있는지 확인)
                    y = np.cross(stickpoints - start, dest - start)
                    y = list(map(np.linalg.norm, y))
                    j = y.index(min(y))
                    # 만약 충분히 가깝다면
                    if min(y) < 0.01:
                        # 그리고 경로 반대 방향에 있는게 아니라면 저장(cross는 반대방향에 있어도 값이 작아지기 때문에)
                        if np.dot(stickpoints[j] - start, dest - start) > 0:
                            points.append([stickpoints[j], 0, j])
                    # 만나는 점이 없다면 그냥 끝
                    if not points:
                        break
                    # points에 시작지점과의 거리를 추가해서 dist에 저장하는 부분
                    for j in points:
                        dist.append([np.linalg.norm(j[0] - start), j])
                    dist.sort()
                    # 있다면 최근거리 점 갖기
                    closest = dist[0]
                    
                    #만약 최근거리 점의 굴절율이 0이 아니라면(점을 만났을때 0이라고 했기 때문에 아니라면 수조임)
                    if closest[1][1] != 0:
                        
                        # 현재 진행 방향
                        original_direction = (dest - start)
                        # 수조면의 정보
                        normal = closest[1][2]
                        new_index = closest[1][1]
                        # 수조 안에서 바깥으로 나가는 상황이라면 노말벡터 방향 바꿔주고 다음 index가 공기의 것임
                        if np.dot(normal, original_direction) > 0:
                            normal = -normal
                            new_index = 1
                        # 각도 계산
                        v, vnorm = -original_direction, np.linalg.norm(-original_direction)
                        w, wnorm = normal, np.linalg.norm(normal)
                        theta1 = np.degrees(np.arccos(np.dot(v, w) / (vnorm * wnorm)))
                        if (current_index * np.sin(np.radians(theta1)) / new_index) > 1:
                            # 전반사 조건
                            theta_diff = 2 * theta1 - 180
                            new_index = current_index
                        else:
                            #전반사가 아닌 조건
                            theta2 = np.degrees(np.arcsin(current_index * np.sin(np.radians(theta1)) / new_index))
                            theta_diff = theta1 - theta2

                        # 굴절을 위한 축
                        axis = np.cross(original_direction, -normal)


                        #굴절시키기
                        # 새로운 경로의 도착점, 시작지점과 index를 저장하고 이 경로로 같은 과정 반복
                        glMatrixMode(GL_MODELVIEW)
                        glPushMatrix()
                        glLoadIdentity()
                        glRotatef(theta_diff, axis[0], axis[1], axis[2])
                        dest = np.matmul(np.append(dest - closest[1][0], [0]), glGetDoublev(GL_MODELVIEW_MATRIX))[:3] + closest[1][0]
                        glPopMatrix()
                        
                        
                        start = closest[1][0]
                        current_index = new_index
                        
                    else:
                        
                        #만약 점과 만났다면(굴절된 경로에 점이 있다면) 원래 목적지 방향(우리가 바라보는 방향)에 dot 표시
                        drawingdestpoints.append(original_dest)
                        break
            #굴절된 점 그리기
            glColor3f(0, 1, 0)
            glPointSize(2)
            glBegin(GL_POINTS)
            for i in drawingdestpoints:
                glVertex3f(i[0], i[1], i[2])
            glEnd()

        else:

            #3d 아닌 다른 화면들을 위한 굴절되지 않은 선 그리기
            glBegin(GL_LINES)
            for i in range(len(stickpoints) - 1):
                glVertex3f(stickpoints[i][0], stickpoints[i][1], stickpoints[i][2])
                glVertex3f(stickpoints[i + 1][0], stickpoints[i + 1][1], stickpoints[i + 1][2])
            glEnd()
                    
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for obj in SubWindow.obj_list:
            #Change the color of selected objects as blue.
            if type(obj) == Sujo:
                if obj.id in self.selected:
                    glColor4f(1, 0, 0, 0.5)
                else:
                    glColor4f(0, 0, 0.2, 0.5)
                obj.draw()


    def drawPickingScene(self):
        """
        Function related to object picking scene drawing.\n
        DO NOT MODIFY THIS.
        """
        glutSetWindow(self.id)

        glClearColor(0.5, 0.5, 0.5, 0.2)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.pickingShader)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(self.projectionMat.T)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.viewMat.T)

        # an object is recognized by its id encoded by unique color
        for obj in SubWindow.obj_list:
            r = (obj.id & 0x000000FF) >> 0
            g = (obj.id & 0x0000FF00) >> 8
            b = (obj.id & 0x00FF0000) >> 16
            glUniform4f(self.pickingColor, r / 255.0, g / 255.0, b / 255.0, 1.0)
            obj.draw()

    def mouse(self, button, state, x, y):
        #We use conversed x and y in mouse events
        cx, cy = self.conversion(x), self.conversion(y)
        """
        Mouse callback function.
        """
        # button macros: GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON
        print(f"Display #{self.id} mouse press event: button={button}, state={state}, x={x}, y={y}")

        #When you turn the mouse wheel, the scale variable is adjusted and do projection to reflect it.
        if button == 3 and state == 0:
            if SubWindow.scale < 1:
                SubWindow.scale = SubWindow.scale + 0.1
            else:
                SubWindow.scale = min(SubWindow.scale + 1, 5)
            self.projection()
        if button == 4 and state == 0:
            if SubWindow.scale <= 1:
                SubWindow.scale = max(0.1, SubWindow.scale - 0.1)
            else:
                SubWindow.scale = SubWindow.scale - 1
            self.projection()

        #When you right-click, the coordinates are saved because camera translation is performed.
        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            if self.id == 5:
                self.currentX = x
                self.currentY = y

        #When left down, store starting coordinates and set moving state as "Selecting".
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.moving = "Selecting"
            self.start = [cx, cy]
            if self.id == 5:
                self.currentX = cx
                self.currentY = cy

        #When left up, check the state if it is "Selecting", which means we do not moved mouse. If not, go to for others.
        if button == GLUT_LEFT_BUTTON and state == GLUT_UP:
            self.rotating = False
            #When control key is pressed, we added picked object's id to selected objects
            #If not pressed, clear the selected objects and add current one.
            #If there is no object on mouse, just clear selected objects.
            if self.moving == "Selecting":
                obj_id = self.pickObject(x, y)
                if obj_id != 0xFFFFFF:
                    print(f"{obj_id} selected")
                    if glutGetModifiers() & GLUT_ACTIVE_CTRL:
                        self.selected.add(obj_id)
                    else:
                        self.selected.clear()
                        self.selected.add(obj_id)
                else:
                    print("Nothing selected")
                    self.selected.clear()
            else:
            #If mouse has moved, we need to do transformations for every selected objects.
                for sel_id in self.selected:
                    for obj in self.obj_list:
                        if sel_id == obj.id:
                            #Do different transformations according to the axis of windows. (by checking SubWindow.id)
                            #If moving state is "Rotation", rotate objects.
                            #When we drag mouse to downside, it rotates counter-clockwise. Upside, clockwise.
                            #Amount is decided by my own discretion
                            if self.moving == "Rotation":
                                if self.id == 2:
                                    obj.xyRotation(200 * self.start[1] - y)
                                if self.id == 3:
                                    obj.yzRotation(-(200 * self.start[1] - y))
                                if self.id == 4:
                                    obj.zxRotation(-(200 * self.start[1] - y))
                            #If moving state is "Scaling", scale objects.
                            #When we drag mouse to right, scale up. Left, scale down. (Horizontal)
                            #When we drag mouse to down, scale up. Up, scale down. (Vertical)
                            #Amount is decided by my own discretion
                            if self.moving == "Scaling":
                                #Horizontal case
                                if abs(self.start[0] - cx) > abs(self.start[1] - cy):
                                    if self.id == 2:
                                        obj.scaling(1 + cx - self.start[0], 1, 1)
                                    if self.id == 3:
                                        obj.scaling(1, 1, 1 + cx - self.start[0])
                                    if self.id == 4:
                                        obj.scaling(1 + cx - self.start[0], 1, 1)
                                #Vertical case
                                else:
                                    if self.id == 2:
                                        obj.scaling(1, 1 + cy - self.start[1], 1)
                                    if self.id == 3:
                                        obj.scaling(1, 1 + cy - self.start[1], 1)
                                    if self.id == 4:
                                        obj.scaling(1, 1, 1 + cy - self.start[1])
                            #If moving state is "Translation", translate objects.
                            if self.moving == "Translation":
                                if self.id == 2:
                                    obj.translation(cx - self.start[0], -(cy - self.start[1]), 0)
                                if self.id == 3:
                                    obj.translation(0, -(cy - self.start[1]), cx - self.start[0])
                                if self.id == 4:
                                    obj.translation(cx - self.start[0], 0, -(cy - self.start[1]))

        if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
            print(f"Add teapot at ({x}, {y})")
            #we need conversion to see appropriate view.
            self.addObject(self.conversion(x - 200), -self.conversion(y - 200))

        self.button = button
        self.modifier = glutGetModifiers()


    def motion(self, x, y):
        """
        Motion (Dragging) callback function.
        """
        print(f"Display #{self.id} mouse move event: x={x}, y={y}, modifer={self.modifier}")

        #Since you are right-clicking and dragging, the camera translation function is executed. Afterwards, the most recent coordinates are saved.
        if self.button == GLUT_RIGHT_BUTTON:

            if self.id == 5:
                self.cameraTranslation(self.currentX, self.currentY, x, y)
                self.currentX = x
                self.currentY = y

        #Since you are left-clicking and dragging, the camera rotation function is executed. Afterwards, the most recent coordinates are saved.
        if self.button == GLUT_LEFT_BUTTON:

            if self.id == 5:
                self.cameraViewRotation(self.currentX, self.currentY, x, y)
                self.currentX = x
                self.currentY = y

            if self.modifier & GLUT_ACTIVE_ALT:
                print("Rotation")
                self.moving = "Rotation"
            elif self.modifier & GLUT_ACTIVE_SHIFT:
                print("Scaling")
                self.moving = "Scaling"
            else:
                print("Translation")
                self.moving = "Translation"


    def pickObject(self, x, y):
        """
        Object picking function.\n
        obj_id can be used to identify which object is clicked, as each object is assigned with unique id.
        """
        self.drawPickingScene()

        data = glReadPixels(x, self.height - y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)

        obj_id = data[0] + data[1] * (2**8) + data[2] * (2**16)

        return obj_id

    def drawAxes(self):
        glPushMatrix()
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0.1, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0.1, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 0.1)
        glColor3f(1, 1, 1)
        glEnd()
        glPopMatrix()

    def addObject(self, x, y):
        # this function should be implemented
        if self.id != 5:
            sujo = Sujo(1.3)
            #Here x and y is conversed mouse coordinates.
            #because first window is x-y window, move (x, y, 0)
            if self.id == 2:
                sujo.translation(x, y, 0)
            #we need to rotate 90 degree from z axis to x axis for second window
            #because second window is z-y window, move (0, y, x)
            if self.id == 3:
                sujo.zxRotation(90)
                sujo.translation(0, y, x)
            #we need to rotate 90 degree from z axis to y axis for second window
            #because third window is x-z window, move conversed (x, 0, z)
            if self.id == 4:
                sujo.yzRotation(-90)
                sujo.translation(x, 0, y)
            # update teapot.mat, etc. to complete your tasks
            SubWindow.obj_list.append(sujo)
            print(sujo.mat)


    #My conversion function. scale down by 200 because one window length is 400 * 400 and our use of length [-1, 1] is 2.
    def conversion(self, val):
        return val / 200
    
    #This is the camera rotation function.
    def cameraViewRotation(self, sx, sy, ex, ey):

        #Set the radius according to the size of the screen.
        #If points are within the radius, calculate z and append it.
        #If not, reduce the number to fit the radius and save it and append 0.
        r = min(self.width, self.height) / 2
        sp = [sx - self.width / 2, -(sy - self.height / 2)]
        ep = [ex - self.width / 2, -(ey - self.height / 2)]
        if sp[0] ** 2 + sp[1] ** 2 > r ** 2:
            sp[0], sp[1] = sp[0] * r / math.sqrt(sp[0] ** 2 + sp[1] ** 2), sp[1] * r / math.sqrt(sp[0] ** 2 + sp[1] ** 2)
            sp.append(0)
        else:
            sp.append(math.sqrt(r ** 2 - sp[0] ** 2 - sp[1] ** 2))
        if ep[0] ** 2 + ep[1] ** 2 > r ** 2:
            ep[0], ep[1] = ep[0] * r / math.sqrt(ep[0] ** 2 + ep[1] ** 2), ep[1] * r / math.sqrt(ep[0] ** 2 + ep[1] ** 2)
            ep.append(0)
        else:
            ep.append(math.sqrt(r ** 2 - ep[0] ** 2 - ep[1] ** 2))

        #Calculate the axis and angle according to the calculated positions of the two points on the hemisphere.
        t = glGetDoublev(GL_MODELVIEW_MATRIX).T
        axis = np.cross(sp, ep) @ t[:3, :3]
        angle = np.dot(sp, ep) / np.linalg.norm(np.dot(sp, ep))

        #If the axis is valid, rotates the viewMat by an angle to the axis.
        if np.linalg.norm(axis) != 0:
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glMultMatrixf(self.viewMat.T)
            glRotatef(angle, axis[0], axis[1], axis[2])
            self.viewMat = glGetDoublev(GL_MODELVIEW_MATRIX).T
            

    #This is the camera translation function.
    def cameraTranslation(self, sx, sy, ex, ey):

        #Get the right vector and up vector for the current screen from modelview matrix.
        t = glGetDoublev(GL_MODELVIEW_MATRIX)

        rightVector = np.array([t[0][0], t[1][0], t[2][0]])
        upVector = np.array([t[0][1], t[1][1], t[2][1]])

        #Calculate the distance you need to translate based on each vector.
        d = rightVector * (ex - sx) / 200.0 + -upVector * (ey - sy) / 200.0

        #Translates the viewMat by an d.
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(self.viewMat.T)
        glTranslatef(d[0], d[1], d[2])
        self.viewMat = glGetDoublev(GL_MODELVIEW_MATRIX).T

        print(self.viewMat)

    #This is the camera projection function.
    def projection(self):

        #If fov is 0, the tan cannot be calculated, so we divide the case.
        if SubWindow.fov != 0:

            #Calculate the distance by scale and fov and create a projection matrix using gluPerspective.
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            distance = SubWindow.scale / math.tan(math.radians(SubWindow.fov / 2.0))
            gluPerspective(SubWindow.fov, self.width / self.height, 0.01, 40.0)
            self.projectionMat = glGetDoublev(GL_PROJECTION_MATRIX).T
            
            #Using modelview matrix, obtain the front and top vectors and the camera position.
            t = self.viewMat.T
            print(t.T)
            frontVector = np.array([t[0][2], t[1][2], t[2][2]])
            upVector = np.array([t[0][1], t[1][1], t[2][1]])
            camera = np.array([t[0][3], t[1][3], t[2][3]])

            #Adjust the new camera position according to the distance and determine the viewing position.
            new_camera = camera + (frontVector * distance)
            new_at = camera + frontVector

            #Look through the camera using gluLookAt.
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(new_camera[0], new_camera[1], new_camera[2], new_at[0], new_at[1], new_at[2], upVector[0], upVector[1], upVector[2])
            self.viewMat = glGetDoublev(GL_MODELVIEW_MATRIX).T

            print(camera, t, self.viewMat)

        else:
            #If fov is 0, glOrtho is used to make it look parallel to the size of the scale.
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(-SubWindow.scale, SubWindow.scale, -SubWindow.scale, SubWindow.scale, -40.0, 40.0)
            self.projectionMat = glGetDoublev(GL_PROJECTION_MATRIX).T

    #When the d key is pressed, the scale is initialized to 1, the rotation is temporarily stored, the position is initialized, and the rotation is applied.
    def keyd(self):
        SubWindow.scale = 1
        t = self.viewMat
        t[:3, 3] = 0
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(t.T)
        self.viewMat = glGetDoublev(GL_MODELVIEW_MATRIX).T
        self.projection()

class Viewer:
    width, height = 800, 800

    def __init__(self):
        pass

    def light(self):
        """
        Light used in the scene.
        """
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        # feel free to adjust light colors
        lightAmbient = [0.5, 0.5, 0.5, 1.0]
        lightDiffuse = [0.5, 0.5, 0.5, 1.0]
        lightSpecular = [0.5, 0.5, 0.5, 1.0]
        lightPosition = [1, 1, -1, 0]  # vector: point at infinity
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular)
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)
        glEnable(GL_LIGHT0)

    def display(self):
        """
        Display callback function for the main window.
        """
        glutSetWindow(self.mainWindow)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1, 1, 1, 0.2)

        glutSwapBuffers()

    def reshape(self, w, h):
        """
        Reshape callback function.\n
        Does notihing as of now.
        """
        print(f"reshape to width: {w}, height: {h}")


    def keyboard(self, key, x, y):
        global startpoint, stickpoints, endpoint
        """
        Keyboard callback function.
        """
        print(f"Display #{glutGetWindow()} keyboard event: key={key}, x={x}, y={y}")
        if glutGetModifiers() & GLUT_ACTIVE_SHIFT:
            print("shift pressed")
        if glutGetModifiers() & GLUT_ACTIVE_ALT:
            print("alt pressed")
        if glutGetModifiers() & GLUT_ACTIVE_CTRL:
            print("ctrl pressed")
        if key == b'\x7f':
            print("del pressed")
            print(SubWindow.selected)
            for i in SubWindow.selected:
                j = 0
                while j < len(SubWindow.obj_list):
                    if SubWindow.obj_list[j].id == i:
                        SubWindow.obj_list.pop(j)
                    else:
                        j += 1
            SubWindow.selected = set()
        #Pressing d initializes translation and scale.
        if key == b'd':
            SubWindow.windows[3].keyd()
        #Pressing 0 initializes fov.
        if key == b'0':
            SubWindow.fov = 0
            SubWindow.windows[3].projection()
        if key == b'q':
            global switch
            switch = not switch
        if key == b'z':
            startpoint = np.array(list(map(float, input().split())))
            print(startpoint)
            stickpoints = np.array([(startpoint * (100 - i) / 100 + endpoint * i / 100) for i in range(101)])
        if key == b'x':
            endpoint = np.array(list(map(float, input().split())))
            print(endpoint)
            stickpoints = np.array([(startpoint * (100 - i) / 100 + endpoint * i / 100) for i in range(101)])
        if key == b'w':
            x = int(input())
            for _ in range(x):
                SubWindow.windows[3].cameraViewRotation(200, 200, 200, 390)
        if key == b'a':
            x = int(input())
            for _ in range(x):
                SubWindow.windows[3].cameraViewRotation(200, 200, 390, 200)
        if key == b's':
            x = int(input())
            for _ in range(x):
                SubWindow.windows[3].cameraViewRotation(200, 200, 200, 10)
        if key == b'd':
            x = int(input())
            for _ in range(x):
                SubWindow.windows[3].cameraViewRotation(200, 200, 10, 200)


    def special(self, key, x, y):
        """
        Special key callback function.
        """
        print(f"Display #{glutGetWindow()} special key event: key={key}, x={x}, y={y}")

        #Adjust the fov using the arrow keys and execute the projection function.
        if key == 101:
            SubWindow.fov = min(90, SubWindow.fov + 5)
            SubWindow.windows[3].projection()
        if key == 103:
            SubWindow.fov = max(0, SubWindow.fov - 5)
            SubWindow.windows[3].projection()

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.mainWindow = glutCreateWindow(b"CS471 Computer Graphics #1")
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)

        # sub-windows
        # xy plane
        SubWindow.windows.append(SubWindow(self.mainWindow, 0, 0, self.width // 2, self.height // 2))
        # zy plane
        SubWindow.windows.append(SubWindow(self.mainWindow, self.width // 2 + 1, 0, self.width // 2, self.height // 2))
        SubWindow.windows[1].viewMat = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        # xz plane
        SubWindow.windows.append(SubWindow(self.mainWindow, 0, self.height // 2 + 1, self.width // 2, self.height // 2))
        SubWindow.windows[2].viewMat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        # 3D
        SubWindow.windows.append(
            SubWindow(self.mainWindow, self.width // 2 + 1, self.height // 2 + 1, self.width // 2, self.height // 2)
        )

        for subWindow in SubWindow.windows:
            glutSetWindow(subWindow.id)
            glutDisplayFunc(subWindow.display)
            glutKeyboardFunc(self.keyboard)
            glutSpecialFunc(self.special)
            glutMouseFunc(subWindow.mouse)
            glutMotionFunc(subWindow.motion)
            #glutPassiveMotionFunc(subWindow.passiveMotion)

            self.light()


        glutMainLoop()


if __name__ == "__main__":

    viewer = Viewer()
    viewer.run()