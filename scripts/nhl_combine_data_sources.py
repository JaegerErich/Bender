# NHL Combine top-10 data by year (2015–2025; 2020–2021 cancelled).
# Sources: topendsports.com, Bleacher Report, NHL.com PDFs.
# Each entry: (year, top10_dict, height_list).
# top10_dict: field_name -> [(player_name, value), ...]
# height_list: [(player_name, height_inches), ...]

def _ft_in_to_in(ft: int, inch: float) -> float:
    return ft * 12 + inch


# ---------- 2015 (topendsports) ----------
TOP10_2015 = {
    "standing_long_jump_in": [
        ("Brendan Guhle", 122), ("Mackenzie Blackwood", 118.9), ("Lawson Crouse", 116.1),
        ("Austin Wagner", 115.4), ("Chris Martenet", 115.0), ("Jack Eichel", 115.0),
        ("Zachary Senyshyn", 115.0), ("Colin White", 113.4), ("Ryan Gropp", 112.2), ("Travis Konecny", 111.0),
    ],
    "vertical_jump_in": [
        ("Austin Wagner", 28.74), ("Ryan Gropp", 27.44), ("MacKenzie Blackwood", 27.28),
        ("Brendan Guhle", 26.77), ("Jack Eichel", 26.18), ("Colin White", 25.00),
        ("Erik Foley", 24.64), ("Thomas Chabot", 24.02), ("Karch Bachman", 23.74), ("Gustav Bouramman", 23.70),
    ],
    "pull_ups": [
        ("Dennis Yan", 14), ("Travis Konecny", 13), ("Erik Foley", 12), ("Kyle Connor", 12),
        ("Brendan Guhle", 12), ("Nathan Noel", 12), ("Brendan Warren", 12), ("Jordan Greenway", 12),
        ("Felix Sandstrom", 12), ("Jesse Gabrielle", 11),
    ],
    "bench_press": [  # reps at 70-80% BW
        ("Jesse Gabrielle", 20), ("Erik Foley", 17), ("Connor Hobbs", 17), ("Jack Eichel", 16),
        ("Travis Konecny", 16), ("Evgeny Svechnikov", 16), ("Brendan Guhle", 15), ("Deven Sideroff", 15),
        ("Kyle Connor", 15), ("Noah Hanifin", 15),
    ],
    "pro_agility_left_sec": [
        ("Colin White", 4.34), ("Blake Speers", 4.38), ("Zachary Senyshyn", 4.42), ("Karch Bachman", 4.43),
        ("Connor McDavid", 4.43), ("Tyler Soy", 4.46), ("Brendan Guhle", 4.47), ("Jack Eichel", 4.48),
        ("Mitchell Stephens", 4.48), ("Callum Booth", 4.5),
    ],
    "pro_agility_right_sec": [
        ("Colin White", 4.21), ("Jack Eichel", 4.24), ("Blake Speers", 4.29), ("Erik Foley", 4.38),
        ("Karch Bachman", 4.39), ("Brendan Guhle", 4.39), ("Tyler Soy", 4.4), ("Mathieu Joseph", 4.42),
        ("Dylan Strome", 4.47),
    ],
    "wingspan_in": [
        ("Daniel Vladar", 84.00), ("Callum Booth", 81.00), ("Yakov Trenin", 80.75),
        ("Jordan Greenway", 80.50), ("Brandon Carlo", 80.50), ("Adin Hill", 80.25),
        ("Paul Bittner", 80.00), ("Chris Martenet", 80.00), ("Erik Cernak", 79.25), ("Roope Hintz", 78.50),
    ],
}
HEIGHT_2015 = [
    ("Chris Martenet", _ft_in_to_in(6, 7)), ("Daniel Vladar", _ft_in_to_in(6, 5.25)),
    ("Brandon Carlo", _ft_in_to_in(6, 5)), ("Jordan Greenway", _ft_in_to_in(6, 4.75)),
]

# ---------- 2016 (topendsports) ----------
TOP10_2016 = {
    "standing_long_jump_in": [
        ("Julien Gauthier", 120.0), ("Jakob Chychrun", 118.5), ("Dylan Wells", 116.3),
        ("Trent Frederic", 114.5), ("Pierre-Luc Dubois", 114.3), ("Jonathan Ang", 114.0),
        ("Givani Smith", 113.8), ("Luke Kunin", 113.5), ("Boris Katchouk", 113.5), ("Vojtech Budik", 113.3),
    ],
    "vertical_jump_in": [
        ("Luke Kunin", 28.98), ("Julien Gauthier", 26.69), ("Chad Krys", 25.59), ("Mitchell Mattson", 25.59),
        ("Noah Gregor", 25.28), ("Jakob Chychrun", 25.28), ("Jordan Kyrou", 24.61), ("Max Zimmer", 24.17),
        ("Wade Allison", 23.74), ("Luke Green", 23.66),
    ],
    "pull_ups": [
        ("Jonathan Ang", 15), ("Sam Steel", 13), ("Carter Hart", 12), ("Clayton Keller", 12),
        ("Pierre-Luc Dubois", 11), ("Jack Kopacka", 11), ("Rasmus Asplund", 11), ("Noah Gregor", 11),
        ("Griffin Luce", 11), ("Nicholas Caamano", 11),
    ],
    "bench_press": [
        ("Samuel Girard", 15), ("Vojtech Budik", 14), ("Tyson Jost", 14), ("Carl Grundstrom", 14),
        ("German Rubtsov", 14), ("Graham McPhee", 14), ("Wade Allison", 13), ("Artur Kayumov", 13),
        ("Dillon Dube", 13), ("Matt Filipe", 13),
    ],
    "pro_agility_left_sec": [
        ("Matthew Cairns", 4.30), ("Luke Green", 4.31), ("Jordan Kyrou", 4.35), ("Cameron Morrison", 4.40),
        ("Joseph Raaymakers", 4.42), ("Jonathan Ang", 4.47), ("Auston Matthews", 4.48), ("Max Zimmer", 4.48),
        ("Connor Bunnaman", 4.49), ("Pierre-Luc Dubois", 4.50),
    ],
    "pro_agility_right_sec": [
        ("Luke Green", 4.19), ("Matthew Cairns", 4.33), ("Jordan Kyrou", 4.33), ("Pierre-Luc Dubois", 4.33),
        ("Luke Kunin", 4.37), ("Trent Frederic", 4.41), ("Joseph Raaymakers", 4.43), ("Max Zimmer", 4.43),
        ("Matt Filipe", 4.44), ("Ty Ronning", 4.46),
    ],
    "wingspan_in": [
        ("Logan Stanley", 82.75), ("Brett Murray", 81.00), ("James Greenway", 80.75),
        ("Givani Smith", 80.25), ("Riley Tufte", 80.00), ("Keaton Middleton", 79.50), ("Logan Brown", 79.50),
        ("Nathan Bastian", 79.00), ("Joseph Raaymakers", 78.25),
    ],
}

# ---------- 2017 (topendsports; names "Last, First" -> First Last) ----------
TOP10_2017 = {
    "standing_long_jump_in": [
        ("Joshua Norris", 118), ("Ian Scott", 117.5), ("Morgan Geekie", 114.5), ("Morgan Frost", 114.3),
        ("Isaac Ratcliffe", 113.5), ("Dayton Rasmussen", 113.3), ("Henri Jokiharju", 111.8),
        ("Timothy Liljegren", 111.5), ("Maxime Comtois", 109.5), ("Cale Makar", 109.3),
    ],
    "vertical_jump_in": [
        ("Joshua Norris", 26.19), ("Dayton Rasmussen", 25.56), ("Morgan Frost", 24.63),
        ("Morgan Geekie", 24.40), ("Cale Makar", 23.69), ("Alex Formenton", 23.63),
        ("Ian Scott", 23.45), ("Henri Jokiharju", 22.84), ("Filip Chytil", 22.62), ("Robin Salo", 22.54),
    ],
    "pull_ups": [
        ("Jack Rathbone", 13), ("Nico Hischier", 13), ("Antoine Morand", 13), ("Mario Ferraro", 12),
        ("Zach Gallant", 12), ("Kailer Yamamoto", 12), ("Jonah Gadjovich", 11), ("Morgan Geekie", 11),
        ("Aleksi Heponiemi", 11), ("Clayton Phillips", 11),
    ],
    "bench_press": [
        ("Dayton Rasmussen", 17), ("Josh Brook", 16), ("Jake Leschyshyn", 16), ("Joshua Norris", 15),
        ("Jack Rathbone", 15), ("Antoine Morand", 15), ("Jonah Gadjovich", 15), ("Mason Shaw", 15),
        ("Lane Zablocki", 15), ("D'artagnan Joly", 14),
    ],
    "pro_agility_left_sec": [
        ("Joshua Norris", 4.19), ("Kailer Yamamoto", 4.38), ("Juuso Valimaki", 4.45),
        ("Scott Reedy", 4.46), ("Mario Ferraro", 4.46), ("Morgan Geekie", 4.46), ("Callan Foote", 4.50),
        ("Cale Makar", 4.52), ("Isaac Ratcliffe", 4.55), ("Thomas Miller", 4.55),
    ],
    "pro_agility_right_sec": [
        ("Joshua Norris", 4.19), ("Kailer Yamamoto", 4.20), ("Cale Makar", 4.39), ("Scott Reedy", 4.41),
        ("Morgan Geekie", 4.41), ("Mario Ferraro", 4.42), ("Maxime Comtois", 4.46), ("Ivan Lodnia", 4.46),
        ("Pierre-Olivier Joseph", 4.48), ("Jack Studnicka", 4.51),
    ],
    "wingspan_in": [
        ("Isaac Ratcliffe", 81.29), ("Keith Petruzzelli", 79.50), ("Michael Rasmussen", 79.25),
        ("Zach Gallant", 79.00), ("Ukko-Pekka Luukkonen", 78.75), ("Jarret Tyszka", 78.75),
        ("Kale Howarth", 78.75), ("Jacob Paquette", 78.74), ("Ian Scott", 78.50), ("Eemeli Rasanen", 78.50),
    ],
}

# ---------- 2018 (topendsports) ----------
TOP10_2018 = {
    "standing_long_jump_in": [
        ("Liam Foudy", 118.8), ("Jacob Ingham", 115.0), ("Nils Lundkvist", 114.5),
        ("Martin Fehervary", 113.5), ("K'Andre Miller", 113.5), ("Jack Gorniak", 111.8),
        ("Carter Robertson", 111.5), ("Olivier Rodrigue", 111.5), ("Jakub Lauko", 111.3),
        ("Noah Dobson", 110.3),
    ],
    "vertical_jump_in": [
        ("Liam Foudy", 27.65), ("Nils Lundkvist", 26.70), ("Martin Fehervary", 24.10),
        ("K'Andre Miller", 23.98), ("Gabriel Fortier", 23.82), ("Jack Gorniak", 23.71),
        ("Jakub Lauko", 23.56), ("Barrett Hayton", 23.31), ("Jordan Harris", 22.78), ("Jacob Ingham", 22.63),
    ],
    "pull_ups": [
        ("Jacob Bernard-Docker", 15), ("Evan Bouchard", 14), ("Jordan Harris", 14),
        ("Gabriel Fortier", 13), ("Martin Fehervary", 13), ("Jay O'Brien", 13),
        ("Mitchell Hoelsher", 13), ("Matej Pekar", 13), ("Benoit-Olivier Groulx", 13), ("Blake McLaughlin", 13),
    ],
    "bench_power_watts_per_kg": [
        ("Rasmus Kupari", 8.25), ("Ty Emberson", 8.03), ("Jack Gorniak", 7.74), ("Liam Foudy", 7.72),
        ("Martin Fehervary", 7.59), ("Kevin Bahl", 7.30), ("Oliver Wahlstrom", 7.25),
        ("Andrei Svechnikov", 7.25), ("Jay O'Brien", 7.21), ("Mitchell Hoelscher", 7.11),
    ],
    "pro_agility_left_sec": [
        ("Jonathan Tychonick", 4.2), ("Liam Foudy", 4.2), ("Jacob Bernard-Docker", 4.3), ("Jack Gorniak", 4.3),
        ("Aidan Dudas", 4.3), ("Xavier Bouchard", 4.4), ("Jack McBain", 4.4), ("Jacob Ingham", 4.4),
        ("Noah Dobson", 4.4), ("Martin Fehervary", 4.4),
    ],
    "pro_agility_right_sec": [
        ("Jonathan Tychonick", 4.2), ("Liam Foudy", 4.3), ("Jack Gorniak", 4.3), ("Andrei Svechnikov", 4.3),
        ("Jacob Bernard-Docker", 4.4), ("Aidan Dudas", 4.4), ("Xavier Bouchard", 4.4),
        ("Sampo Ranta", 4.4), ("Jake Wise", 4.4), ("Gabriel Fortier", 4.4),
    ],
    "wingspan_in": [
        ("Kevin Bahl", 81.75), ("Curtis Douglas", 81.25), ("Jack McBain", 81.25),
        ("Mattias Samuelsson", 80.75), ("Kevin Mandolese", 80.25), ("K'Andre Miller", 79.75),
        ("Jakub Skarek", 79.00), ("Jacob Ingham", 79.00), ("Keegan Karki", 78.75), ("Xavier Bernard", 78.50),
    ],
}

# ---------- 2019 (topendsports; Cole Caulfield = Caufield) ----------
TOP10_2019 = {
    "standing_long_jump_in": [
        ("Jayden Struble", 117.8), ("Spencer Knight", 117.0), ("Hunter Jones", 116.5),
        ("Judd Caulfield", 116.3), ("John Beecher", 115.3), ("Cade Webber", 113.5),
        ("Samuel Bolduc", 113.3), ("Dylan Cozens", 112.0), ("Henry Thrun", 111.8), ("Drew Helleson", 111.5),
    ],
    "vertical_jump_force_plate_in": [
        ("Marshall Warren", 26.05), ("Hugo Alnefelt", 24.28), ("Jayden Struble", 23.88),
        ("Samuel Bolduc", 23.82), ("Spencer Knight", 23.58), ("Cole Caufield", 23.08),
        ("Judd Caulfield", 22.98), ("Yegor Spiridonov", 22.22), ("Alexander Campbell", 22.18), ("Albin Grewe", 22.14),
    ],
    "pull_ups": [
        ("Cole Caufield", 16), ("Nils Hoglander", 16), ("Artemi Kniazev", 16), ("Mikhail Abramov", 16),
        ("Peyton Krebs", 15), ("Trevor Janicke", 15), ("Nicholas Robertson", 15), ("Henry Thrun", 14),
        ("Samuel Bolduc", 13), ("Alexander Campbell", 13),
    ],
    "bench_power_watts_per_kg": [
        ("Jayden Struble", 9.42), ("Nils Hoglander", 7.98), ("Trevor Janicke", 7.82),
        ("Matthew Robertson", 7.36), ("Samuel Bolduc", 7.20), ("Henry Thrun", 7.11),
        ("Cole Caufield", 7.07), ("Case McCarthy", 6.84), ("Harrison Blaisdell", 6.83), ("Michael Vukojevic", 6.60),
    ],
    "pro_agility_left_sec": [
        ("Alex Newhook", 4.25), ("Kaedan Korczak", 4.26), ("Spencer Knight", 4.29),
        ("Harrison Blaisdell", 4.31), ("Matvey Guskov", 4.31), ("Cade Webber", 4.33),
        ("Artemi Kniazev", 4.35), ("Nils Hoglander", 4.36), ("Nathan Legare", 4.36), ("Brayden Tracey", 4.36),
    ],
    "pro_agility_right_sec": [
        ("Samuel Bolduc", 4.28), ("Spencer Knight", 4.32), ("Nicholas Robertson", 4.32),
        ("Matvey Guskov", 4.34), ("Nils Hoglander", 4.34), ("Jakob Pelletier", 4.34),
        ("Kirby Dach", 4.35), ("Brett Leason", 4.35), ("Henry Thrun", 4.38), ("Samuel Poulin", 4.38),
    ],
    "wingspan_in": [
        ("Alex Vlasic", 80.75), ("Cade Webber", 79.75), ("Moritz Seider", 79.50),
        ("Raphael Lavoie", 79.00), ("Mads Sogaard", 79.00), ("Hunter Jones", 78.75),
        ("Brett Leason", 78.50), ("Philip Broberg", 78.25), ("Kaedan Korczak", 78.00), ("Nolan Foote", 78.00),
    ],
}

# ---------- 2022 (topendsports) ----------
TOP10_2022 = {
    "standing_long_jump_in": [
        ("Maveric Lamoureux", 117.5), ("Isaiah George", 115.0), ("Owen Beck", 114.5),
        ("Cutter Gauthier", 112.0), ("Dylan James", 112.0), ("Fabian Wagner", 112.0),
        ("Frank Nazar", 112.0), ("Jack Hughes", 111.5), ("Marek Hejduk", 110.0), ("Matyas Sapovaliv", 110.0),
    ],
    "vertical_jump_force_plate_in": [
        ("Isaiah George", 21.73), ("Frank Nazar", 21.69), ("Fabian Wagner", 21.29),
        ("Marek Hejduk", 21.28), ("Aleksanteri Kaskimaki", 21.28), ("Maveric Lamoureux", 20.82),
        ("Jack Hughes", 19.94), ("Cutter Gauthier", 19.78), ("Owen Beck", 19.63), ("Logan Cooley", 19.43),
    ],
    "pull_ups": [
        ("Jack Hughes", 19), ("Lane Hutson", 18), ("Jake Karabela", 18), ("Julian Lutz", 17),
        ("Matthew Poitras", 17), ("Owen Beck", 16), ("Hunter Haight", 16), ("Isaac Howard", 16),
        ("Jack Devine", 15), ("Cutter Gauthier", 15),
    ],
    "bench_power_watts_per_kg": [
        ("Filip Mesar", 9.43), ("Brennan Ali", 9.07), ("Liam Arnsby", 9.06), ("Seamus Casey", 9.03),
        ("Jake Karabela", 8.79), ("Jack Hughes", 8.49), ("Lane Hutson", 8.13), ("Sam Rinzel", 8.09),
        ("Jackson Edward", 8.03), ("Christian Kyrou", 7.99),
    ],
    "pro_agility_left_sec": [
        ("Michael Fisher", 4.33), ("Julian Lutz", 4.38), ("Jack Hughes", 4.40), ("Brennan Ali", 4.42),
        ("Maveric Lamoureux", 4.43), ("Matyas Sapovaliv", 4.44), ("Christian Kyrou", 4.44),
        ("Cutter Gauthier", 4.46), ("Marco Kasper", 4.49), ("Shane Wright", 4.49),
    ],
    "pro_agility_right_sec": [
        ("Michael Fisher", 4.25), ("Christian Kyrou", 4.38), ("Owen Beck", 4.42),
        ("Maveric Lamoureux", 4.48), ("Jake Karabela", 4.54), ("Jack Hughes", 4.55),
        ("Brennan Ali", 4.59), ("Ryan Greene", 4.59), ("Isaiah George", 4.59), ("Cameron Lund", 4.61),
    ],
    "wingspan_in": [
        ("Noah Warren", 82.00), ("Lian Bichsel", 80.00), ("Jack Sparkes", 80.00),
        ("Maveric Lamoureux", 79.75), ("Tyler Brennan", 78.75), ("Topias Leinonen", 78.50),
        ("Conor Geekie", 77.75), ("Filip Bystedt", 77.75), ("Alexander Suzdalev", 77.50), ("Adam Ingram", 77.00),
    ],
}

# ---------- 2023 (topendsports; strip " (Team)" from names) ----------
TOP10_2023 = {
    "standing_long_jump_in": [
        ("Charlie Stramel", 118.0), ("Samuel Honzek", 117.5), ("Cameron Allen", 117.0),
        ("Tom Willander", 116.5), ("Beau Akey", 116.0), ("Nick Lardis", 114.5),
        ("Maxim Strbak", 112.8), ("Carey Terrance", 112.0), ("David Reinbacher", 112.0), ("Nate Danielson", 111.5),
    ],
    "vertical_jump_force_plate_in": [
        ("Nick Lardis", 25.49), ("Cameron Allen", 23.81), ("Colby Barlow", 22.96),
        ("Caden Price", 22.83), ("Charlie Stramel", 22.71), ("Nate Danielson", 22.05),
        ("Carey Terrance", 22.00), ("Jesse Nurmi", 21.99), ("Beau Akey", 21.80), ("Oliver Moore", 21.70),
    ],
    "pull_ups": [
        ("Nick Lardis", 15), ("Cameron Allen", 14), ("Danny Nelson", 14), ("Connor Bedard", 14),
        ("Bradly Nadeau", 14), ("Brad Gardiner", 13), ("Jaden Lipinski", 13), ("Nate Danielson", 13),
        ("Oliver Moore", 13),
    ],
    "bench_power_watts_per_kg": [
        ("Axel Sandin Pellikka", 8.39), ("Charlie Stramel", 7.92), ("Mathieu Cataford", 7.61),
        ("Lukas Dragicevic", 7.58), ("Ryan Leonard", 7.13), ("Cameron Allen", 7.09),
        ("Jonathan Castagna", 7.07), ("Bradly Nadeau", 7.06), ("Brad Gardiner", 7.04), ("Nate Danielson", 7.03),
    ],
    "pro_agility_left_sec": [
        ("Easton Cowan", 4.04), ("Jonathan Castagna", 4.09), ("Oliver Moore", 4.13),
        ("Brandon Svoboda", 4.21), ("Jayden Perron", 4.22), ("Brad Gardiner", 4.24),
        ("William Whitelaw", 4.24), ("Ethan Gauthier", 4.26), ("Beau Akey", 4.26), ("Andrew Gibson", 4.26),
    ],
    "pro_agility_right_sec": [
        ("Easton Cowan", 4.09), ("Jonathan Castagna", 4.16), ("Brady Cleveland", 4.24),
        ("Samuel Honzek", 4.32), ("Scott Ratzlaff", 4.32), ("Brandon Svoboda", 4.33),
        ("Brad Gardiner", 4.35), ("Beau Akey", 4.35), ("Andrew Gibson", 4.36), ("Hunter Brzustewicz", 4.36),
    ],
    "wingspan_in": [
        ("Michael Hrabal", 82.51), ("Matteo Mann", 80.50), ("Carsen Musser", 79.75),
        ("Zachary Nehring", 79.50), ("Jakub Stancl", 78.50), ("Larry Keenan", 78.50),
        ("Brady Cleveland", 78.25), ("Brandon Svoboda", 78.00), ("Jaden Lipinski", 77.75), ("Anton Wahlberg", 77.50),
    ],
}

# ---------- 2024 (topendsports) ----------
TOP10_2024 = {
    "standing_long_jump_in": [
        ("EJ Emery", 123.0), ("AJ Spellacy", 119.3), ("Mikhail Yegorov", 115.8),
        ("Jack Pridham", 114.5), ("Alexis Bernier", 114.5), ("Dean Letourneau", 114.5),
        ("Linus Eriksson", 113.8), ("Javon Moore", 113.0), ("Aatos Koivu", 113.0), ("Dominik Badinka", 113.0),
    ],
    "vertical_jump_force_plate_in": [
        ("EJ Emery", 27.23), ("Jett Luchanko", 24.63), ("Veeti Vaisanen", 24.22),
        ("Will Skahan", 23.88), ("Stian Solberg", 23.84), ("Mikhail Yegorov", 23.27),
        ("Jack Pridham", 22.83), ("Carson Wetsch", 22.76), ("AJ Spellacy", 22.64), ("Artyom Levshunov", 22.64),
    ],
    "pull_ups": [
        ("Zeev Buium", 16), ("Lukas Fischer", 15), ("Colton Roberts", 13), ("Jack Pridham", 13),
        ("Cole Beaudoin", 13), ("Ollie Josephson", 13), ("Tij Iginla", 13), ("Kamil Bednarik", 12),
        ("Ethan Procyszyn", 12), ("Kevin He", 12),
    ],
    "bench_power_watts_per_kg": [
        ("Cole Beaudoin", 8.71), ("Lukas Fischer", 8.26), ("John Mustard", 7.78),
        ("Cayden Lindstrom", 7.69), ("Sebastian Soini", 7.54), ("Raoul Boilard", 7.42),
        ("Sam O'Reilly", 6.95), ("Carson Wetsch", 6.92), ("Sam Dickinson", 6.92), ("Nathan Villeneuve", 6.88),
    ],
    "pro_agility_left_sec": [
        ("Stian Solberg", 4.12), ("Kamil Bednarik", 4.25), ("John Mustard", 4.26),
        ("Terik Parascak", 4.28), ("Michael Hage", 4.29), ("Charlie Elick", 4.33),
        ("Sam O'Reilly", 4.33), ("Konsta Helenius", 4.34), ("Lucas Pettersson", 4.34), ("Jack Pridham", 4.34),
    ],
    "pro_agility_right_sec": [
        ("Stian Solberg", 4.08), ("Jett Luchanko", 4.23), ("EJ Emery", 4.23), ("AJ Spellacy", 4.24),
        ("Terik Parascak", 4.25), ("Sam O'Reilly", 4.25), ("Alexis Bernier", 4.28), ("Jack Pridham", 4.29),
        ("Michael Brandsegg-Nygard", 4.29), ("Ethan Procyszyn", 4.30),
    ],
    "wingspan_in": [
        ("Dean Letourneau", 83.25), ("Gabriel Eliasson", 81.0), ("EJ Emery", 80.75),
        ("Jesse Pulkkinen", 80.75), ("Eriks Mateiko", 80.25), ("Tomas Lavoie", 80.0),
        ("Adam Kleber", 80.0), ("Adam Jecho", 79.5), ("Will Skahan", 79.0), ("Sam Dickinson", 78.75),
    ],
}

# 2025 is in fetch_nhl_combine_data.py (TOP10_2025, HEIGHT_2025)

ALL_YEARS_DATA = [
    (2015, TOP10_2015, HEIGHT_2015, "nhl_combine_2015"),
    (2016, TOP10_2016, [], "nhl_combine_2016"),
    (2017, TOP10_2017, [], "nhl_combine_2017"),
    (2018, TOP10_2018, [], "nhl_combine_2018"),
    (2019, TOP10_2019, [], "nhl_combine_2019"),
    (2022, TOP10_2022, [], "nhl_combine_2022"),
    (2023, TOP10_2023, [], "nhl_combine_2023"),
    (2024, TOP10_2024, [], "nhl_combine_2024"),
]
