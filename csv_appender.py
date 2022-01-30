from flask import Flask, render_template, request
from csv import writer

App = Flask(__name__)


@App.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@App.route('/operation_result/', methods=['POST'])
def operation_result():
    error = None
    first_input = request.form["Length"]
    second_input = request.form["Height"]
    third_input = request.form["Width"]
    try:
        input1, input2, input3 = float(first_input), float(second_input), float(third_input)
        volume = input1 * input2 * input3
        dimensions = [input1, input2, input3]

        def csv_appender(dimensions):
            with open('dimensions.csv', 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow(dimensions)
        csv_appender(dimensions)
        return render_template('index.html', input1=input1, input2=input2, input3=input3, volume=volume, input_success=True)
    except volume == 0:
        return render_template('index.html', input1=input1, input2=input2, input3=input3, volume=volume, result="Bad Input", input_success=False, error_message="volume can't be zero")


if __name__ == '__main__':
    App.debug = True
    App.run()
