import numpy as np
import matplotlib.pyplot as plt
import tkinter as Tk
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class Shubert_Piyavskii():
    def __init__(self, function, lower_bound, upper_bound, lipschitz_constant):
        self.function = function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lipschitz_constant = lipschitz_constant

        self.intersection_xs = np.array([])
        self.intersection_ys = np.array([])
        self.intersections = []

        self.sampled_xs = np.array([self.lower_bound, self.upper_bound])
        self.sampled_ys = np.array([self.function(self.lower_bound), self.function(self.upper_bound)])

        self.fig = plt.Figure(dpi=100)
        self.ax = self.fig.add_subplot(xlim=(self.lower_bound - .1 * (self.upper_bound - self.lower_bound), self.upper_bound + .1 * (self.upper_bound - self.lower_bound)), ylim=(-10, 2)) 
        self.lw = 2

        self.root = Tk.Tk()

    def find_intersection(self, x_lower, x_upper, y_lower, y_upper):
        t = (self.lipschitz_constant * x_upper - self.lipschitz_constant * x_lower + y_lower - y_upper) / (2 * self.lipschitz_constant)
        intersection = np.array([x_lower + t, y_lower - self.lipschitz_constant*t])

        return intersection
    
    def add_intersection(self, intersection):
        self.intersections.append(intersection)

        self.intersection_xs = np.append(self.intersection_xs, [intersection[0]])
        self.intersection_ys = np.append(self.intersection_ys, [intersection[1]])

        inds = np.argsort(self.intersection_xs)
        self.intersection_xs = self.intersection_xs[inds]
        self.intersection_ys = self.intersection_ys[inds]

    def add_sample(self, x):
        # Append Sample
        self.sampled_xs = np.append(self.sampled_xs, x)
        self.sampled_ys = np.append(self.sampled_ys, self.function(x))

        # Sort Samples
        inds = np.argsort(self.sampled_xs)
        self.sampled_xs = self.sampled_xs[inds]
        self.sampled_ys = self.sampled_ys[inds]

        # Add New Intersections
        index = np.where(self.sampled_xs == x)[0][0]
        self.add_intersection(self.find_intersection(self.sampled_xs[index-1], x, self.sampled_ys[index-1], self.sampled_ys[index]))

        index = np.where(self.sampled_xs == x)[0][0]
        self.add_intersection(self.find_intersection(x, self.sampled_xs[index+1], self.sampled_ys[index], self.sampled_ys[index+1]))

        # Delete Old Intersections
        index = np.where(self.sampled_xs == x)[0][0]
        self.intersection_xs = np.delete(self.intersection_xs, index)
        self.intersection_ys = np.delete(self.intersection_ys, index)

    def plot_function(self, num_points=1000):
        x_pts = [self.lower_bound + (self.upper_bound - self.lower_bound)*(i/num_points) for i in range(int(num_points))]
        y_pts = [self.function(x) for x in x_pts]

        plt.plot(x_pts, y_pts)

    def plot_intersection(self, intersection, left_x, right_x):
        plt.plot([left_x, intersection[0]], [self.function(left_x), intersection[1]], "orange")
        plt.plot([intersection[0], right_x], [intersection[1], self.function(right_x)], "orange")

    def iterate(self, iterations=10):
        intersection = self.find_intersection(self.sampled_xs[0], self.sampled_xs[1], self.sampled_ys[0], self.sampled_ys[1])
        self.add_intersection(intersection)

        for _ in range(iterations):
            argmin_intersection = np.argmin(self.intersection_ys)
            self.add_sample(self.intersection_xs[argmin_intersection])

        # Plot
        self.plot_function()

        for i in range(len(self.intersection_xs)):
            self.plot_intersection([self.intersection_xs[i], self.intersection_ys[i]], self.sampled_xs[i], self.sampled_xs[i+1])

    def animate(self, intersections, left_x, right_x):
        function_samples = 5000
        frames_per_section = 200
        num_sections = len(intersections) * 2 + 1
        num_frames = int(frames_per_section * num_sections) # index "i" will go up to num_frames - 1

        lines_per_intersection = 3
        num_lines = len(intersections) * lines_per_intersection + 2
        lines = []

        # Line 0, 1 = initial lines
        # Line 3n + 2 = sample
        # Line 3n + 3 and 3n + 4 = left and right lines

        for line in range(num_lines):
            if ((line - 2) % lines_per_intersection == 0):
                line, = self.ax.plot([], [], lw=self.lw, color='b', linestyle='dashed', zorder=0)
            elif (line == 0 or line == 1):
                line, = self.ax.plot([], [], lw=self.lw, color='b', zorder=0)
            else:
                line, = self.ax.plot([], [], lw=self.lw, color='b', zorder=0)
            lines.append(line)

        x = np.array([self.lower_bound + (self.upper_bound - self.lower_bound)*(i/function_samples) for i in range(int(function_samples))])
        y = np.array([self.function(x) for x in [self.lower_bound + (self.upper_bound - self.lower_bound)*(i/function_samples) for i in range(int(function_samples))]])
        self.ax.plot(x, y, lw=2, color='orange')

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        def animate(i):
            # Based on "i" calculate the section number and the intersection number
            section_number = int(i / frames_per_section)
            intersection_number = int((i - frames_per_section) / (2 * frames_per_section))

            if (i < frames_per_section):
                x = np.array([left_x, (i+1)/frames_per_section * (intersections[0][0] - left_x) + left_x])
                y = np.array([self.function(left_x), (i+1)/frames_per_section * (intersections[0][1] - self.function(left_x)) + self.function(left_x)])
                lines[0].set_data(x, y)

                x = np.array([(i+1)/frames_per_section * (intersections[0][0] - right_x) + right_x, right_x])
                y = np.array([(i+1)/frames_per_section * (intersections[0][1] - self.function(right_x)) + self.function(right_x), self.function(right_x)])
                lines[1].set_data(x, y)

                if ((i / (frames_per_section)) == 1):
                    # time.sleep(5)
                    pass
            else:
                if (section_number % 2 == 1):
                    x = np.array([intersections[intersection_number][0], intersections[intersection_number][0]])
                    y = np.array([intersections[intersection_number][1], (i-section_number*frames_per_section)/frames_per_section * (self.function(intersections[intersection_number][0]) - intersections[intersection_number][1]) + intersections[intersection_number][1]])
                    lines[int(lines_per_intersection*intersection_number)+2].set_data(x, y) # Draw Vertical Sample Point
                else:
                    x_0l = intersections[intersection_number][0]
                    y_0l = ((i+1)-section_number*frames_per_section)/frames_per_section * (self.function(intersections[intersection_number][0]) - intersections[intersection_number][1]) + intersections[intersection_number][1]
                    x_1l = (1 / (2 * -self.lipschitz_constant)) * (-self.lipschitz_constant * intersections[intersection_number][0] + -self.lipschitz_constant * x_0l + y_0l - intersections[intersection_number][1])
                    y_1l = -self.lipschitz_constant * (x_1l - intersections[intersection_number][0]) + intersections[intersection_number][1]
                    xl = np.array([x_0l, x_1l])
                    yl = np.array([y_0l, y_1l])

                    x_0r = intersections[intersection_number][0]
                    y_0r = ((i+1)-section_number*frames_per_section)/frames_per_section * (self.function(intersections[intersection_number][0]) - intersections[intersection_number][1]) + intersections[intersection_number][1]
                    x_1r = (1 / (2 * self.lipschitz_constant)) * (self.lipschitz_constant * intersections[intersection_number][0] + self.lipschitz_constant * x_0r + y_0r - intersections[intersection_number][1])
                    y_1r = self.lipschitz_constant * (x_1r - intersections[intersection_number][0]) + intersections[intersection_number][1]
                    xr = np.array([x_0r, x_1r])
                    yr = np.array([y_0r, y_1r])

                    if (i % frames_per_section == 0):
                        for line_num in range(int(lines_per_intersection*intersection_number)+2):
                            x_data = lines[line_num].get_xdata()
                            y_data = lines[line_num].get_ydata()

                            if ((np.isclose(x_data, intersections[intersection_number][0], atol=0.01).any()) and (np.isclose(y_data, intersections[intersection_number][1], atol=0.01).any())):
                                
                                if (x_data[0] + x_data[1] < 2*intersections[intersection_number][0]):
                                    self.left_line_num = line_num
                                    if (x_data[0] < intersections[intersection_number][0]):
                                        self.keep_left_x = x_data[0]
                                        self.keep_left_y = y_data[0]
                                    else:
                                        self.keep_left_x = x_data[1]
                                        self.keep_left_y = y_data[1]
                                if (x_data[0] + x_data[1] > 2*intersections[intersection_number][0]):
                                    self.right_line_num = line_num
                                    if (x_data[0] > intersections[intersection_number][0]):
                                        self.keep_right_x = x_data[0]
                                        self.keep_right_y = y_data[0]
                                    else:
                                        self.keep_right_x = x_data[1]
                                        self.keep_right_y = y_data[1]

                    lines[int(self.left_line_num)].set_data([self.keep_left_x, x_1l],[self.keep_left_y, y_1l])
                    lines[int(self.right_line_num)].set_data([self.keep_right_x, x_1r],[self.keep_right_y, y_1r])

                    lines[int(lines_per_intersection*intersection_number)+3].set_data(xl, yl) # Draw New Left Line
                    lines[int(lines_per_intersection*intersection_number)+4].set_data(xr, yr) # Draw New Right Line

                    x = np.array([intersections[intersection_number][0], intersections[intersection_number][0]])
                    y = np.array([self.function(intersections[intersection_number][0]), (i-section_number*frames_per_section)/frames_per_section * (self.function(intersections[intersection_number][0]) - intersections[intersection_number][1]) + intersections[intersection_number][1]])
                    lines[int(lines_per_intersection*intersection_number)+2].set_data(x, y) # Erase Vertical Sample Point
            return lines

        anim = animation.FuncAnimation(self.fig, animate, frames=num_frames, interval=.5, blit=True, repeat=False)
        Tk.mainloop()

if (__name__ == "__main__"):
    def test_func(x):
        return .5 * np.cos(x) + np.sin(x)

    obj = Shubert_Piyavskii(test_func, -1, 5, 3)
    obj.iterate(7)
    # plt.show()
    obj.animate(obj.intersections, obj.lower_bound, obj.upper_bound)