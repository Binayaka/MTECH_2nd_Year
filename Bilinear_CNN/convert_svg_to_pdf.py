"""this will convert svg files to pdf files """
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

def run():
    """this will run the main function """
    drawing = svg2rlg('model_plot.svg')
    renderPDF.drawToFile(drawing, 'model_plot.pdf')

if __name__ == '__main__':
    run()