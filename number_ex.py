import pytesseract as pt

print(pt.image_to_string("390test.png",'eng','1234567890'))