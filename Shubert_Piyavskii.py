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

        self.sampled_xs = np.array([self.lower_bound, self.upper_bound])
        self.sampled_ys = np.array([self.function(self.lower_bound), self.function(self.upper_bound)])

        self.fig = plt.Figure(dpi=100)
        self.ax = self.fig.add_subplot(xlim=(self.lower_bound - .1 * (self.upper_bound - self.lower_bound), self.upper_bound + .1 * (self.upper_bound - self.lower_bound)), ylim=(-10, 2)) 
        self.num_points = 999

        self.root = Tk.Tk()

    def find_intersection(self, x_lower, x_upper, y_lower, y_upper):
        t = (self.lipschitz_constant * x_upper - self.lipschitz_constant * x_lower + y_lower - y_upper) / (2 * self.lipschitz_constant)
        intersection = np.array([x_lower + t, y_lower - self.lipschitz_constant*t])

        return intersection
    
    def add_intersection(self, intersection):
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

    
    def animate(self, intersection, left_x, right_x):
        x = np.array([self.lower_bound + (self.upper_bound - self.lower_bound)*(i/self.num_points) for i in range(int(self.num_points))])
        y = np.array([self.function(x) for x in [self.lower_bound + (self.upper_bound - self.lower_bound)*(i/self.num_points) for i in range(int(self.num_points))]])
        
        line0, = self.ax.plot(x, y, lw=2, color='orange')
        line1, = self.ax.plot([], [], lw=2, color='b')
        line2, = self.ax.plot([], [], lw=2, color='b')
        line3, = self.ax.plot([], [], lw=2, color='b', linestyle='dashed')
        line4, = self.ax.plot([], [], lw=2, color='b')
        line5, = self.ax.plot([], [], lw=2, color='b')
        line6, = self.ax.plot([], [], lw=3, color='w')
        line7, = self.ax.plot([], [], lw=3, color='w')
        line8, = self.ax.plot([], [], lw=3, color='w')

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        num_sections = 3
        points_per_section = int(self.num_points / num_sections)
        slope1 = -(intersection[1] - self.function(left_x)) / (intersection[0] - left_x)
        slope2 = -slope1
        delta = 0.02
        
        def animate(i):
            if ((i / points_per_section) <= 1):
                x = np.array([left_x, i/points_per_section * (intersection[0] - left_x) + left_x])
                y = np.array([self.function(left_x), i/points_per_section * (intersection[1] - self.function(left_x)) + self.function(left_x)])
                line1.set_data(x, y)

                x = np.array([i/points_per_section * (intersection[0] - right_x) + right_x, right_x])
                y = np.array([i/points_per_section * (intersection[1] - self.function(right_x)) + self.function(right_x), self.function(right_x)])
                line2.set_data(x, y)

                if ((i / points_per_section) == 1):
                    time.sleep(1)
            
            elif ((i / points_per_section) <= 2):
                x = np.array([intersection[0], intersection[0]])
                y = np.array([intersection[1], (i-points_per_section)/points_per_section * (self.function(intersection[0]) - intersection[1]) + intersection[1]])
                line3.set_data(x, y)

                if ((i / points_per_section) == 2):
                    time.sleep(0)

            elif ((i / points_per_section) <= 3):
                x_0 = intersection[0]
                y_0 = (i-2*points_per_section)/points_per_section * (self.function(intersection[0]) - intersection[1]) + intersection[1]
                x_1 = (1 / (2 * slope1)) * (slope1 * intersection[0] + slope1 * x_0 + y_0 - intersection[1])
                y_1 = slope1 * (x_1 - intersection[0]) + intersection[1]

                x = np.array([x_0, x_1])
                y = np.array([y_0, y_1])
                
                line4.set_data(x, y)

                delta_x = delta * (x_1 - intersection[0])
                delta_y = slope1 * delta_x
                line6.set_data([intersection[0], x_1 - delta_x], [intersection[1], y_1 - delta_y] )

                x_0 = intersection[0]
                y_0 = (i-2*points_per_section)/points_per_section * (self.function(intersection[0]) - intersection[1]) + intersection[1]
                x_1 = (1 / (2 * slope2)) * (slope2 * intersection[0] + slope2 * x_0 + y_0 - intersection[1])
                y_1 = slope2 * (x_1 - intersection[0]) + intersection[1]

                x = np.array([x_0, x_1])
                y = np.array([y_0, y_1])

                line5.set_data(x, y)

                delta_x = delta * (x_1 - intersection[0])
                delta_y = slope2 * delta_x
                line7.set_data([intersection[0], x_1 - delta_x], [intersection[1], y_1 - delta_y] )

                delta_y = delta * (self.function(intersection[0]) - intersection[1])
                line8.set_data([intersection[0], x_0], [intersection[1], y_0 - delta_y])

            return line1, line2, line3, line4, line5, line6, line7, line8

        anim = animation.FuncAnimation(self.fig, animate, frames=self.num_points, interval=2, blit=True, repeat=False)

        Tk.mainloop()

    def animate_sample_line(self, intersection):
        line3, = self.ax.plot([], [], lw=2, color='b', linestyle='dashed')

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        def init():
            line3.set_data([], [])
            return line3,

        def animate(i):
            x = np.array([intersection[0], intersection[0]])
            y = np.array([intersection[1], i/self.num_points * (intersection[0] - intersection[1]) + intersection[1]])
            line3.set_data(x, y)

            return line3,

        anim = animation.FuncAnimation(self.fig, animate, init_func=init,frames=self.num_points, interval=2, blit=True, repeat=False)

        Tk.mainloop()

if (__name__ == "__main__"):
    def test_func(x):
        # return x ** 3
        return .5 * np.cos(x) + np.sin(x)

    obj = Shubert_Piyavskii(test_func, -1, 5, 3)
    intersection = obj.find_intersection(obj.lower_bound, obj.upper_bound, obj.function(obj.lower_bound), obj.function(obj.upper_bound))
    obj.animate(intersection, obj.lower_bound, obj.upper_bound)
    # obj.animate([1, -3], -2, 4)
    # obj.animate_sample_line([2, -4])
    # obj.iterate(0)
    # plt.show()