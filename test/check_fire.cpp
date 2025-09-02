#include "tasks/packet_typedef.hpp"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#ifdef Success
    #undef Success
#endif
#include <chrono>
#include <iostream>
#include <thread>
#include <wust_vl/wust_vl.hpp>

int main(int argc, char** argv) {
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        std::abort();
    });

    SerialDriver serial;
    SerialDriver::SerialPortConfig cfg { 115200,
                                         8,
                                         boost::asio::serial_port_base::parity::none,
                                         boost::asio::serial_port_base::stop_bits::one,
                                         boost::asio::serial_port_base::flow_control::none };
    serial.init_port("/dev/ttyACM0", cfg);
    serial.set_error_callback([&](const boost::system::error_code& ec) {
        WUST_ERROR("serial") << "serial error: " << ec.message();
    });
    serial.start();

    bool run = true;
    SignalHandler sig;
    sig.start([&] { run = false; });

    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) {
        std::cerr << "Cannot open X display\n";
        return -1;
    }
    Window root = DefaultRootWindow(dpy);

    bool last_pressed = false;

    while (!sig.shouldExit() && run) {
        unsigned int mask;
        Window ret_root, ret_child;
        int root_x, root_y, win_x, win_y;

        XQueryPointer(dpy, root, &ret_root, &ret_child, &root_x, &root_y, &win_x, &win_y, &mask);

        bool current_pressed = (mask & Button1Mask) != 0;
        bool fire = false;

        if (!last_pressed && current_pressed) {
            std::cout << "fire firefirefirefirefirefirefire" << std::endl;
            fire = true;
        } else if (last_pressed && !current_pressed) {
        }

        last_pressed = current_pressed;

        SendRobotCmdData send_data;
        send_data.cmd_ID = ID_ROBOT_CMD;
        send_data.appear = false;
        send_data.fire = fire;
        serial.write(std::move(toVector(send_data)));
        std::cout << "send" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    XCloseDisplay(dpy);
    return 0;
}
