temperatur = 22

# Einfache Bedingung
if temperatur > 25:
    print("Heute ist es heiß!")
elif temperatur > 15:
    print("Angenehme Temperatur")
else:
    print("Heute ist es kalt")

# Mehrere Bedingungen
alter = 20
if alter >= 18 and alter < 65:
    print("Arbeitsfähiges Alter")

# Praktisches Beispiel
noten = [1.7, 2.3, 1.0, 3.7, 2.0]
print("Notenbewertung:")
for note in noten:
    if note <= 1.5:
        print(f"{note} - Sehr gut!")
    elif note <= 2.5:
        print(f"{note} - Gut")
    elif note <= 3.5:
        print(f"{note} - Befriedigend")
    elif note <= 4.0:
        print(f"{note} - Ausreichend")
    else:
        print(f"{note} - Nicht bestanden")