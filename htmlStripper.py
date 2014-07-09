from html.parser import HTMLParser
import urllib.request as urllib

class MyHTMLParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.data = ''
		self.recording = 0
	def handle_starttag(self, tag, attrs):
		if tag != 'div':
			return
		if self.recording:
			self.recording += 1
			return
		for name, value in attrs:
			if name == 'class' and value == 'productDescriptionWrapper':
				self.recording = 1
			else:
				return
			
	def handle_endtag(self, tag):
		if tag == 'div' and self.recording:
			self.recording -= 1
		
	def handle_data(self, data):
		if self.recording:
			self.data = self.data + data


def readAmazonReviews(isbn):
	url = 'http://www.amazon.com/dp/product-description/' + isbn + '/'
	usocket = urllib.urlopen(url)
	content = usocket.read()
	usocket.close()
	
	contentStr = content.decode("utf-8", "ignore")
	parser = MyHTMLParser()
	parser.feed(contentStr)
	return parser.data
	
	
