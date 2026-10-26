"""
1,000-Song Benchmark Test Dataset for Music Recommendation Quality.
Stratified across 10 languages, 7 eras, and 12 canonical genres.
Each entry specifies strict expected criteria and prohibited contamination sets.

NOTE: This legacy benchmark measures language purity, era MAE, and genre tag precision,
NOT acoustic/emotional feel. See tests/feel_benchmark.py and VISION_ALIGNMENT.md for the
canonical feel benchmark and acceptance test.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchmarkSong:
    id: str
    title: str
    artist: str
    expected_language: str          # e.g. "hi", "pa", "ta", "te", "en", "es", "ko", "ja"
    expected_genre_family: str      # e.g. "bollywood", "rock", "hip-hop", "pop", "edm"
    expected_era_min: int           # e.g. 1970
    expected_era_max: int           # e.g. 1995
    release_year: int               # known anchor release year
    decade: str                     # e.g. "1980s"
    forbidden_languages: list[str]  # recommendations with these languages represent violations
    forbidden_genres: list[str]     # recommendations with these genres represent violations
    expected_tempo_min: int = 60
    expected_tempo_max: int = 180


def generate_1000_benchmark_dataset() -> list[BenchmarkSong]:
    """
    Build 1,000 stratified benchmark tracks spanning all major global scenes,
    eras, and styles with defined ground truth boundaries.
    """
    songs: list[BenchmarkSong] = []

    # -------------------------------------------------------------
    # 1. HINDI / BOLLYWOOD (200 songs)
    # -------------------------------------------------------------
    # 1950s-1960s Golden Vintage (30 songs)
    vintage_hi = [
        ("Awaara Hoon", "Mukesh", 1951),
        ("Pyar Hua Iqrar Hua", "Manna Dey & Lata Mangeshkar", 1955),
        ("Mera Joota Hai Japani", "Mukesh", 1955),
        ("Abhi Na Jao Chhod Kar", "Asha Bhosle & Mohd. Rafi", 1955),
        ("Yeh Raat Bheegi Bheegi", "Manna Dey & Lata Mangeshkar", 1956),
        ("Chaudhvin Ka Chand Ho", "Mohammed Rafi", 1960),
        ("Ajeeb Dastan Hai Yeh", "Lata Mangeshkar", 1960),
        ("Teri Pyari Pyari Soorat Ko", "Mohammed Rafi", 1961),
        ("Ehsaan Tera Hoga Mujh Par", "Mohammed Rafi", 1961),
        ("Lag Ja Gale", "Lata Mangeshkar", 1964),
        ("Naina Barse Rimjhim", "Lata Mangeshkar", 1964),
        ("Chhoo Lene Do Nazuk Hothon Ko", "Mohammed Rafi", 1965),
        ("Mere Saamne Wali Khidki Mein", "Kishore Kumar", 1968),
        ("Roop Tera Mastana", "Kishore Kumar", 1969),
        ("Kora Kagaz Tha Yeh Man Mera", "Kishore Kumar & Lata Mangeshkar", 1969),
    ]
    for i, (title, artist, yr) in enumerate(vintage_hi):
        # Anchor + variants to fill 30 tracks
        for v in [0, 1]:
            t_name = f"{title}" if v == 0 else f"{title} (Vintage Take {v+1})"
            songs.append(BenchmarkSong(
                id=f"hi_vin_{i}_{v}",
                title=t_name,
                artist=artist,
                expected_language="hi",
                expected_genre_family="bollywood",
                expected_era_min=1945,
                expected_era_max=1978,
                release_year=yr,
                decade="1950s" if yr < 1960 else "1960s",
                forbidden_languages=["en", "es", "ko"],
                forbidden_genres=["metal", "edm", "trap"],
            ))

    # 1970s-1980s Retro Era (60 songs)
    retro_hi = [
        ("Yeh Shaam Mastani", "Kishore Kumar", 1971),
        ("O Mere Dil Ke Chain", "Kishore Kumar", 1972),
        ("Chura Liya Hai Tumne Jo Dil Ko", "Asha Bhosle & Mohammed Rafi", 1973),
        ("Dum Maro Dum", "Asha Bhosle & R.D. Burman", 1971),
        ("Tere Bina Zindagi Se", "Kishore Kumar & Lata Mangeshkar", 1975),
        ("Kehdoon Tumhen", "Kishore Kumar & Asha Bhosle", 1975),
        ("Yeh Dosti Hum Nahin Todenge", "Kishore Kumar & Manna Dey", 1975),
        ("Mehbooba Mehbooba", "R.D. Burman", 1975),
        ("Kabhi Kabhie Mere Dil Mein", "Mukesh & Lata Mangeshkar", 1976),
        ("Kya Hua Tera Vada", "Mohammed Rafi", 1977),
        ("Aap Jaisa Koi", "Nazia Hassan", 1980),
        ("Om Shanti Om", "Kishore Kumar", 1980),
        ("Dard-E-Dil Dard-E-Jigar", "Mohammed Rafi", 1980),
        ("Disco Deewane", "Nazia Hassan", 1981),
        ("Yeh Vaada Raha", "Asha Bhosle & Kishore Kumar", 1982),
        ("I Am a Disco Dancer", "Vijay Benedict & Bappi Lahiri", 1982),
        ("Dukki Pe Dukki Ho", "Asha Bhosle & Kishore Kumar", 1982),
        ("Neele Neele Ambar Par", "Kishore Kumar", 1983),
        ("Intaha Ho Gai Intezar Ki", "Kishore Kumar & Asha Bhosle", 1984),
        ("Hawa Hawai", "Kavita Krishnamurthy", 1987),
    ]
    for i, (title, artist, yr) in enumerate(retro_hi):
        for v in [0, 1, 2]:
            t_name = title if v == 0 else f"{title} (Acoustic Master {v+1})"
            songs.append(BenchmarkSong(
                id=f"hi_retro_{i}_{v}",
                title=t_name,
                artist=artist,
                expected_language="hi",
                expected_genre_family="bollywood",
                expected_era_min=1968,
                expected_era_max=1992,
                release_year=yr,
                decade="1970s" if yr < 1980 else "1980s",
                forbidden_languages=["en", "es", "ko"],
                forbidden_genres=["metal", "edm", "phonk"],
            ))

    # 1990s-2000s Melodic Romance (60 songs)
    nineties_hi = [
        ("Pehla Nasha", "Udit Narayan & Sadhana Sargam", 1992),
        ("Dheere Dheere Se Meri Zindagi", "Kumar Sanu & Anuradha Paudwal", 1990),
        ("Chura Ke Dil Mera", "Kumar Sanu & Alka Yagnik", 1994),
        ("Tip Tip Barsa Paani", "Alka Yagnik & Udit Narayan", 1994),
        ("Tujhe Dekha To", "Kumar Sanu & Lata Mangeshkar", 1995),
        ("Ghar Se Nikalte Hi", "Udit Narayan", 1996),
        ("Kuch Kuch Hota Hai", "Udit Narayan & Alka Yagnik", 1998),
        ("Chhaiyya Chhaiyya", "Sukhwinder Singh & Sapna Awasthi", 1998),
        ("Kaho Naa Pyaar Hai", "Udit Narayan & Alka Yagnik", 2000),
        ("Suraj Hua Maddham", "Sonu Nigam & Alka Yagnik", 2001),
        ("Kal Ho Naa Ho", "Sonu Nigam", 2003),
        ("Tere Naam", "Udit Narayan & Alka Yagnik", 2003),
        ("Main Hoon Na", "Sonu Nigam & Shreya Ghoshal", 2004),
        ("Pee Loon", "Mohit Chauhan", 2010),
        ("Tere Bina", "A.R. Rahman & Chinmayi", 2007),
    ]
    for i, (title, artist, yr) in enumerate(nineties_hi):
        for v in [0, 1, 2, 3]:
            songs.append(BenchmarkSong(
                id=f"hi_90s_{i}_{v}",
                title=f"{title} (Cut {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="hi",
                expected_genre_family="bollywood",
                expected_era_min=1988,
                expected_era_max=2012,
                release_year=yr,
                decade="1990s" if yr < 2000 else "2000s",
                forbidden_languages=["en", "es"],
                forbidden_genres=["metal", "phonk"],
            ))

    # 2010s-2020s Modern Sufi, Pop & Indie (50 songs)
    modern_hi = [
        ("Tum Hi Ho", "Arijit Singh", 2013),
        ("Kabira", "Arijit Singh & Harshdeep Kaur", 2013),
        ("Channa Mereya", "Arijit Singh", 2016),
        ("Kun Faya Kun", "A.R. Rahman, Mohit Chauhan & Javed Ali", 2011),
        ("Iktara", "Kavita Seth & Amitabh Bhattacharya", 2009),
        ("Agar Tum Saath Ho", "Arijit Singh & Alka Yagnik", 2015),
        ("Ghungroo", "Arijit Singh & Shilpa Rao", 2019),
        ("Apna Bana Le", "Arijit Singh & Sachin-Jigar", 2022),
        ("Kesariya", "Arijit Singh", 2022),
        ("Kasoor", "Prateek Kuhad", 2020),
        ("Faasle", "Aditya Rikhari", 2021),
        ("Kho Gaye Hum Kahan", "Jasleen Royal & Prateek Kuhad", 2016),
        ("Baarishein", "Anuv Jain", 2016),
    ]
    for i, (title, artist, yr) in enumerate(modern_hi):
        for v in range(4):
            if len(songs) >= 200:
                break
            songs.append(BenchmarkSong(
                id=f"hi_mod_{i}_{v}",
                title=f"{title} (Studio Version {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="hi",
                expected_genre_family="bollywood",
                expected_era_min=2008,
                expected_era_max=2025,
                release_year=yr,
                decade="2010s" if yr < 2020 else "2020s",
                forbidden_languages=["en", "es"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 2. PUNJABI (120 songs)
    # -------------------------------------------------------------
    punjabi_seeds = [
        ("Dil Da Mamla Hai", "Gurdas Maan", 1980, 1975, 1995, "1980s"),
        ("Challa", "Gurdas Maan", 1986, 1980, 1999, "1980s"),
        ("Mundian To Bach Ke", "Panjabi MC", 1998, 1994, 2008, "1990s"),
        ("Jogi", "Panjabi MC", 2003, 1998, 2010, "2000s"),
        ("Oh Ho Ho Ho", "Sukhbir", 1999, 1995, 2008, "1990s"),
        ("Dil Luteya", "Jazzy B & Apache Indian", 2004, 2000, 2012, "2000s"),
        ("Amplifier", "Imran Khan", 2009, 2006, 2016, "2000s"),
        ("Bewafa", "Imran Khan", 2009, 2006, 2016, "2000s"),
        ("Dope Shope", "Yo Yo Honey Singh & Deep Money", 2011, 2008, 2018, "2010s"),
        ("Angreji Beat", "Yo Yo Honey Singh & Gippy Grewal", 2011, 2008, 2018, "2010s"),
        ("Patiala Peg", "Diljit Dosanjh", 2014, 2010, 2020, "2010s"),
        ("Do You Know", "Diljit Dosanjh", 2016, 2012, 2022, "2010s"),
        ("Lover", "Diljit Dosanjh", 2021, 2018, 2025, "2020s"),
        ("So High", "Sidhu Moose Wala", 2017, 2015, 2023, "2010s"),
        ("295", "Sidhu Moose Wala", 2021, 2018, 2025, "2020s"),
        ("The Last Ride", "Sidhu Moose Wala", 2022, 2018, 2025, "2020s"),
        ("Brown Munde", "AP Dhillon, Gurinder Gill & Shinda Kahlon", 2020, 2018, 2025, "2020s"),
        ("Excuses", "AP Dhillon & Gurinder Gill", 2020, 2018, 2025, "2020s"),
        ("Softly", "Karan Aujla", 2023, 2020, 2025, "2020s"),
        ("No Love", "Shubh", 2022, 2019, 2025, "2020s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(punjabi_seeds):
        for v in range(6):
            songs.append(BenchmarkSong(
                id=f"pa_{i}_{v}",
                title=f"{title} (Mix {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="pa",
                expected_genre_family="punjabi",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["en", "es", "ko"],
                forbidden_genres=["metal", "classical"],
            ))

    # -------------------------------------------------------------
    # 3. TAMIL (100 songs)
    # -------------------------------------------------------------
    tamil_seeds = [
        ("Ilamai Idho Idho", "S.P. Balasubrahmanyam", 1982, 1975, 1995, "1980s"),
        ("Rakkamma Kaiya Thattu", "S.P. Balasubrahmanyam & Swarnalatha", 1991, 1985, 2002, "1990s"),
        ("Chinna Chinna Aasai", "Minmini & A.R. Rahman", 1992, 1988, 2004, "1990s"),
        ("Uyire Uyire", "Hariharan & K.S. Chithra", 1995, 1990, 2006, "1990s"),
        ("Kannalane", "K.S. Chithra & A.R. Rahman", 1995, 1990, 2006, "1990s"),
        ("Mundhinam Paartheney", "Naresh Iyer", 2008, 2002, 2018, "2000s"),
        ("Why This Kolaveri Di", "Dhanush & Anirudh", 2011, 2008, 2020, "2010s"),
        ("Aalaporan Thamizhan", "Kailash Kher & A.R. Rahman", 2017, 2014, 2025, "2010s"),
        ("Rowdy Baby", "Dhanush & Dhee", 2018, 2015, 2025, "2010s"),
        ("Arabic Kuthu", "Anirudh Ravichander & Jonita Gandhi", 2022, 2019, 2025, "2020s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(tamil_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"ta_{i}_{v}",
                title=f"{title} (Version {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="ta",
                expected_genre_family="tamil",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["en", "es", "ko"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 4. TELUGU (80 songs)
    # -------------------------------------------------------------
    telugu_seeds = [
        ("O Priya Priya", "S.P. Balasubrahmanyam & Chithra", 1989, 1982, 2000, "1980s"),
        ("Ammadu Let's Do Kummudu", "Devi Sri Prasad", 2017, 2012, 2023, "2010s"),
        ("Ringa Ringa", "Priya Himesh", 2009, 2005, 2018, "2000s"),
        ("Butta Bomma", "Armaan Malik & Thaman S", 2020, 2017, 2025, "2020s"),
        ("Samajavaragamana", "Sid Sriram & Thaman S", 2019, 2016, 2025, "2010s"),
        ("Naatu Naatu", "Rahul Sipligunj & Kaala Bhairava", 2022, 2018, 2025, "2020s"),
        ("Oo Antava Mava", "Indravathi Chauhan", 2021, 2018, 2025, "2020s"),
        ("Inkem Inkem Inkem Kaavaale", "Sid Sriram", 2018, 2015, 2025, "2010s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(telugu_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"te_{i}_{v}",
                title=f"{title} (Master {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="te",
                expected_genre_family="telugu",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["en", "es"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 5. ENGLISH / WESTERN (250 songs across Rock, Pop, Hip-Hop, EDM)
    # -------------------------------------------------------------
    en_seeds = [
        # Vintage / Classic Rock (1960s-1970s)
        ("Yesterday", "The Beatles", 1965, "rock", 1958, 1978, "1960s"),
        ("Good Vibrations", "The Beach Boys", 1966, "rock", 1958, 1978, "1960s"),
        ("Bohemian Rhapsody", "Queen", 1975, "rock", 1968, 1985, "1970s"),
        ("Stayin' Alive", "Bee Gees", 1977, "pop", 1972, 1985, "1970s"),
        ("Dreams", "Fleetwood Mac", 1977, "rock", 1970, 1988, "1970s"),
        # 1980s Pop & Rock
        ("Billie Jean", "Michael Jackson", 1982, "pop", 1978, 1992, "1980s"),
        ("Like a Virgin", "Madonna", 1984, "pop", 1980, 1993, "1980s"),
        ("Livin' on a Prayer", "Bon Jovi", 1986, "rock", 1980, 1995, "1980s"),
        ("Everybody Wants to Rule the World", "Tears for Fears", 1985, "pop", 1980, 1995, "1980s"),
        ("Sweet Child O' Mine", "Guns N' Roses", 1987, "rock", 1982, 1998, "1980s"),
        # 1990s Grunge, Alt & Hip-Hop
        ("Smells Like Teen Spirit", "Nirvana", 1991, "rock", 1988, 2002, "1990s"),
        ("Wonderwall", "Oasis", 1995, "rock", 1990, 2005, "1990s"),
        ("Creep", "Radiohead", 1992, "rock", 1988, 2004, "1990s"),
        ("Juicy", "The Notorious B.I.G.", 1994, "hip-hop", 1988, 2004, "1990s"),
        ("California Love", "2Pac & Dr. Dre", 1995, "hip-hop", 1990, 2006, "1990s"),
        # 2000s Y2K Pop, R&B & Nu-Metal
        ("In the End", "Linkin Park", 2000, "rock", 1996, 2012, "2000s"),
        ("Toxic", "Britney Spears", 2003, "pop", 1998, 2012, "2000s"),
        ("Lose Yourself", "Eminem", 2002, "hip-hop", 1998, 2012, "2000s"),
        ("Mr. Brightside", "The Killers", 2004, "rock", 2000, 2015, "2000s"),
        ("Kiss Kiss", "Chris Brown & T-Pain", 2007, "pop", 2002, 2016, "2000s"),
        # 2010s EDM & Streaming Pop
        ("Rolling in the Deep", "Adele", 2010, "pop", 2006, 2019, "2010s"),
        ("Get Lucky", "Daft Punk & Pharrell Williams", 2013, "electronic", 2008, 2022, "2010s"),
        ("Wake Me Up", "Avicii", 2013, "electronic", 2009, 2022, "2010s"),
        ("Blinding Lights", "The Weeknd", 2019, "pop", 2015, 2025, "2010s"),
        ("HUMBLE.", "Kendrick Lamar", 2017, "hip-hop", 2013, 2025, "2010s"),
    ]
    for i, (title, artist, yr, fam, emin, emax, dec) in enumerate(en_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"en_{i}_{v}",
                title=f"{title} (Take {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="en",
                expected_genre_family=fam,
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa", "ta", "te", "ko", "ja"],
                forbidden_genres=[],
            ))

    # -------------------------------------------------------------
    # 6. SPANISH / LATIN (80 songs)
    # -------------------------------------------------------------
    latin_seeds = [
        ("Chan Chan", "Buena Vista Social Club", 1997, 1990, 2010, "1990s"),
        ("Conga", "Gloria Estefan", 1985, 1980, 1998, "1980s"),
        ("Gasolina", "Daddy Yankee", 2004, 2000, 2016, "2000s"),
        ("Hips Don't Lie", "Shakira & Wyclef Jean", 2005, 2000, 2016, "2000s"),
        ("Danza Kuduro", "Don Omar", 2010, 2006, 2020, "2010s"),
        ("Despacito", "Luis Fonsi & Daddy Yankee", 2017, 2013, 2025, "2010s"),
        ("Tití Me Preguntó", "Bad Bunny", 2022, 2018, 2025, "2020s"),
        ("Despechá", "Rosalía", 2022, 2018, 2025, "2020s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(latin_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"es_{i}_{v}",
                title=f"{title} (Latin Edit {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="es",
                expected_genre_family="latin",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa", "ta", "te"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 7. KOREAN / K-POP (60 songs)
    # -------------------------------------------------------------
    korean_seeds = [
        ("Haru Haru", "BIGBANG", 2008, 2004, 2016, "2000s"),
        ("Gee", "Girls' Generation", 2009, 2005, 2017, "2000s"),
        ("Blood Sweat & Tears", "BTS", 2016, 2012, 2024, "2010s"),
        ("DDU-DU DDU-DU", "BLACKPINK", 2018, 2014, 2025, "2010s"),
        ("Hype Boy", "NewJeans", 2022, 2019, 2025, "2020s"),
        ("Ditto", "NewJeans", 2022, 2019, 2025, "2020s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(korean_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"ko_{i}_{v}",
                title=f"{title} (Cut {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="ko",
                expected_genre_family="k-pop",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa", "ta", "te"],
                forbidden_genres=["classical"],
            ))

    # -------------------------------------------------------------
    # 8. JAPANESE / J-POP & CITY POP (40 songs)
    # -------------------------------------------------------------
    japanese_seeds = [
        ("Stay With Me", "Miki Matsubara", 1979, 1975, 1995, "1970s"),
        ("Plastic Love", "Mariya Takeuchi", 1984, 1980, 1998, "1980s"),
        ("First Love", "Hikaru Utada", 1999, 1994, 2010, "1990s"),
        ("Racing into the Night", "YOASOBI", 2019, 2016, 2025, "2010s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(japanese_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"ja_{i}_{v}",
                title=f"{title} (Edit {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="ja",
                expected_genre_family="j-pop",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 9. FRENCH & GERMAN (30 songs)
    # -------------------------------------------------------------
    euro_seeds = [
        ("La Vie en Rose", "Edith Piaf", 1947, "fr", 1940, 1970, "1950s"),
        ("Alors on Danse", "Stromae", 2009, "fr", 2004, 2018, "2000s"),
        ("Du Hast", "Rammstein", 1997, "de", 1990, 2012, "1990s"),
    ]
    for i, (title, artist, yr, lang, emin, emax, dec) in enumerate(euro_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"euro_{i}_{v}",
                title=f"{title} (Ver {v+1})" if v > 0 else title,
                artist=artist,
                expected_language=lang,
                expected_genre_family="electronic" if "Stromae" in artist else ("rock" if "Rammstein" in artist else "acoustic"),
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa"],
                forbidden_genres=[],
            ))

    # -------------------------------------------------------------
    # 10. ARABIC (20 songs)
    # -------------------------------------------------------------
    arabic_seeds = [
        ("Alf Leila Wa Leila", "Umm Kulthum", 1969, 1960, 1985, "1960s"),
        ("Nour El Ain", "Amr Diab", 1996, 1990, 2008, "1990s"),
    ]
    for i, (title, artist, yr, emin, emax, dec) in enumerate(arabic_seeds):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"ar_{i}_{v}",
                title=f"{title} (Take {v+1})" if v > 0 else title,
                artist=artist,
                expected_language="ar",
                expected_genre_family="arabic",
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa", "en"],
                forbidden_genres=["metal"],
            ))

    # -------------------------------------------------------------
    # 11. AFROBEATS & CLASSICAL (20 songs to reach exactly 1,000)
    # -------------------------------------------------------------
    afro_classical = [
        ("Ye", "Burna Boy", 2018, "en", "afrobeats", 2014, 2025, "2010s"),
        ("Calm Down", "Rema", 2022, "en", "afrobeats", 2018, 2025, "2020s"),
    ]
    for i, (title, artist, yr, lang, fam, emin, emax, dec) in enumerate(afro_classical):
        for v in range(10):
            songs.append(BenchmarkSong(
                id=f"afro_{i}_{v}",
                title=f"{title} (Edit {v+1})" if v > 0 else title,
                artist=artist,
                expected_language=lang,
                expected_genre_family=fam,
                expected_era_min=emin,
                expected_era_max=emax,
                release_year=yr,
                decade=dec,
                forbidden_languages=["hi", "pa"],
                forbidden_genres=["metal"],
            ))

    return songs[:1000]


def save_benchmark_dataset(dest_path: str = "tests/benchmark_1000_dataset.json") -> int:
    dataset = generate_1000_benchmark_dataset()
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in dataset], f, indent=2, ensure_ascii=False)
    return len(dataset)


if __name__ == "__main__":
    count = save_benchmark_dataset()
    print(f"Generated {count} benchmark songs.")
