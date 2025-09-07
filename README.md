# Control Package with RPLidar C1 on ROS 2 Humble

Este proyecto integra un RPLidar C1 conectado a una Raspberry Pi 4B usando **ROS 2 Humble**.  
El sistema consta de dos partes:  
1. El driver `sllidar_ros2`, que publica el topic `/scan` con los datos del Lidar.  
2. El paquete `control_pkg`, que procesa esos datos y detecta el objeto más cercano.  

---

## 🚀 Pasos de instalación y uso

### 1. Compilar el workspace
Ubícate en tu workspace (ejemplo: `ros_ws`) y compila:

```bash
cd ~/OpenCV_VC/ros_ws
colcon build
```
Carga las configuraciones:
```bash
source install/setup.bash
```
2. Conectar el Lidar

Conecta el RPLidar C1 por USB a la Raspberry Pi.
Verifica el puerto con:
```bash
ls /dev/ttyUSB*
```
Normalmente será /dev/ttyUSB0.

Dale permisos de acceso:
```bash
sudo chmod 777 /dev/ttyUSB0
```
3. Lanzar el driver del Lidar

Ejecuta el launch del paquete sllidar_ros2:
```bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py
```
Esto empezará a publicar datos en el topic /scan.

Puedes verificarlo con:
```bash
ros2 topic list
ros2 topic echo /scan
```
4. Ejecutar el nodo de procesamiento

En otra terminal, recuerda cargar nuevamente el entorno:
```bash
cd ~/OpenCV_VC/ros_ws
source install/setup.bash
```
Ejecuta tu nodo:

ros2 run control_pkg closest_object
