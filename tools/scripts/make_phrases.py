"""Generates bench/phrases.jsonl, the pre-registered benchmark set (Req 6.1). Run from the repo root.

Checks rule 3 of bench/README.md for the phrase's own words, set sizes, duplicates, and disjointness from the pilot.
"""
import json
import unicodedata
from collections import Counter

P = {
    "places": [
        ("the Leaning Tower of Pisa", "italy italian 意大利 tuscany galileo"),
        ("the Grand Canyon", "arizona colorado 亚利桑那 america usa 美国"),
        ("Machu Picchu", "peru inca incas 秘鲁 andes 印加"),
        ("the Kremlin", "moscow russia russian 莫斯科 俄罗斯 putin soviet"),
        ("Angkor Wat", "cambodia khmer 柬埔寨 temple siem hindu"),
        ("the Golden Gate Bridge", "san francisco california 旧金山 加州 bay"),
        ("Times Square", "york manhattan 纽约 broadway america 美国"),
        ("the Acropolis", "athens greece greek 雅典 希腊 parthenon"),
        ("Stonehenge", "england britain english 英国 druids neolithic salisbury"),
        ("the Dead Sea", "israel jordan 以色列 约旦 salt jerusalem"),
        ("Kilimanjaro", "tanzania africa african 非洲 坦桑尼亚 kenya"),
        ("the Forbidden City", "beijing 北京 china chinese 中国 emperor ming qing"),
        ("the Alps", "switzerland swiss 瑞士 austria mountains skiing france"),
        ("Silicon Valley", "california 加州 google apple tech startups startup stanford"),
        ("the Vatican", "pope catholic rome 罗马 教皇 天主教 italy"),
        ("Chernobyl", "ukraine nuclear 乌克兰 radiation soviet reactor"),
        ("the Panama Canal", "ships atlantic pacific 太平洋 大西洋 suez"),
        ("Hollywood", "movies movie film films 电影 actors california los angeles"),
        ("the Louvre", "paris 巴黎 france 法国 museum mona lisa"),
        ("the Himalayas", "nepal everest tibet india 尼泊尔 珠穆朗玛 西藏"),
    ],
    "people": [
        ("Napoleon Bonaparte", "france french 法国 emperor waterloo corsica"),
        ("Marie Curie", "radium polonium radioactivity physics chemistry nobel 物理 化学"),
        ("Nikola Tesla", "electricity edison electric alternating 爱迪生 inventor musk"),
        ("Ludwig van Beethoven", "mozart bach symphony composer 莫扎特 巴赫 作曲家 交响曲"),
        ("Vincent van Gogh", "painter sunflowers 向日葵 starry impressionist gauguin 画家 monet"),
        ("Mahatma Gandhi", "india indian 印度 independence nehru nonviolence"),
        ("Nelson Mandela", "south africa apartheid 南非 african"),
        ("Genghis Khan", "mongol mongols mongolia 蒙古 empire"),
        ("Confucius", "china chinese 中国 儒家 analects philosopher 论语 laozi 老子"),
        ("Karl Marx", "communism communist engels 共产主义 capital socialism lenin 恩格斯"),
        ("Steve Jobs", "apple iphone 苹果 mac wozniak pixar"),
        ("Elvis Presley", "rock roll singer memphis 猫王 beatles"),
        ("Michael Jordan", "basketball nba bulls chicago 篮球 kobe lebron"),
        ("Cleopatra", "egypt egyptian 埃及 caesar antony pharaoh"),
        ("Sigmund Freud", "psychoanalysis psychology unconscious 心理学 jung vienna 精神分析"),
        ("Alan Turing", "computer computing enigma 计算机 machine codebreaker ai"),
        ("Pablo Picasso", "painter cubism 立体主义 画家 spanish spain guernica"),
        ("Galileo Galilei", "telescope astronomy copernicus 天文学 kepler newton italian"),
        ("Frida Kahlo", "mexico mexican 墨西哥 painter rivera 画家"),
        ("Muhammad Ali", "boxing boxer 拳击 heavyweight tyson frazier"),
    ],
    "events": [
        ("the Cold War", "soviet ussr 苏联 america nuclear 美国 communism berlin"),
        ("the Moon landing", "apollo armstrong nasa 阿波罗 astronaut astronauts"),
        ("the Black Death", "plague 鼠疫 pandemic bubonic medieval epidemic"),
        ("the fall of the Berlin Wall", "germany german 德国 1989 soviet cold communism"),
        ("the Industrial Revolution", "steam factories factory britain 蒸汽 industrialization coal"),
        ("the September 11 attacks", "terrorism terrorist terrorists 恐怖 qaeda bin laden twin towers"),
        ("the Renaissance", "italy florence leonardo michelangelo 意大利 medici"),
        ("the Great Depression", "1929 unemployment economic recession roosevelt crash 经济"),
        ("the American Civil War", "lincoln slavery confederate confederacy union 林肯 grant gettysburg"),
        ("the Russian Revolution", "lenin bolshevik bolsheviks 列宁 tsar soviet 苏联 stalin"),
        ("the Spanish flu", "1918 pandemic epidemic virus 病毒"),
        ("the Holocaust", "nazi nazis hitler jews jewish 纳粹 希特勒 auschwitz 犹太"),
        ("the Boston Tea Party", "colonists revolution america american british tax taxes 美国"),
        ("the Cultural Revolution", "mao 毛泽东 china 中国 red guards"),
        ("the Battle of Waterloo", "napoleon 拿破仑 wellington belgium 1815 french"),
    ],
    "science": [
        ("photosynthesis", "chlorophyll plants plant sunlight 叶绿素 植物 carbon glucose"),
        ("the theory of relativity", "einstein 爱因斯坦 spacetime physics 物理 gravity"),
        ("DNA", "gene genes genetic chromosome 基因 rna double helix 染色体"),
        ("black holes", "hawking gravity event horizon spacetime 引力 astrophysics galaxy 霍金"),
        ("the periodic table", "elements element mendeleev chemistry 化学 元素 atomic"),
        ("plate tectonics", "earthquakes earthquake continents continental 地震 geology volcano"),
        ("vaccines", "immunity immune vaccination virus antibodies 免疫 病毒"),
        ("the Big Bang", "universe cosmology 宇宙 expansion cosmic inflation"),
        ("neurons", "brain synapse synapses 大脑 axon neural nervous"),
        ("climate change", "global warming carbon emissions 全球变暖 碳排放 greenhouse"),
        ("antibiotics", "bacteria bacterial penicillin 细菌 青霉素 infection infections"),
        ("the speed of light", "einstein relativity photons photon physics vacuum"),
        ("volcanoes", "lava magma eruption eruptions 岩浆 喷发"),
        ("the human genome", "dna genes gene 基因 sequencing chromosomes"),
        ("artificial intelligence", "machine learning neural robots 机器学习 深度学习 神经网络"),
    ],
    "culture": [
        ("Harry Potter", "hogwarts rowling wizard magic 魔法 霍格沃茨 hermione dumbledore"),
        ("The Lord of the Rings", "tolkien hobbit frodo gandalf 托尔金 霍比特"),
        ("the Beatles", "lennon mccartney liverpool band 列侬"),
        ("Pokémon", "pikachu nintendo 皮卡丘 任天堂 ash"),
        ("the Bible", "jesus god christian christianity 基督教 耶稣 testament gospel"),
        ("Romeo and Juliet", "shakespeare 莎士比亚 verona tragedy hamlet"),
        ("Game of Thrones", "westeros stark lannister targaryen dragons hbo"),
        ("Super Mario", "nintendo luigi 任天堂 bowser princess peach zelda"),
        ("jazz", "blues swing saxophone trumpet armstrong 布鲁斯 coltrane"),
        ("sushi", "japan japanese 日本 rice sashimi wasabi 生鱼片"),
        ("Christmas", "santa claus december jesus gifts easter reindeer"),
        ("Mickey Mouse", "disney 迪士尼 minnie donald cartoon walt"),
        ("ballet", "dance dancer dancers tchaikovsky 舞蹈 swan nutcracker"),
        ("K-pop", "korea korean 韩国 bts idol idols seoul blackpink"),
        ("the Quran", "islam muslim muslims allah muhammad 伊斯兰 穆斯林"),
    ],
    "indonesian": [
        ("Borobudur", "buddhist buddhism temple java javanese indonesia magelang candi yogyakarta 印尼 佛教"),
        ("Bali", "hindu indonesia 印尼 denpasar ubud beach island kuta"),
        ("Jakarta", "indonesia 印尼 capital ibukota batavia java bandung surabaya"),
        ("Soekarno", "indonesia 印尼 independence merdeka hatta suharto soeharto president presiden"),
        ("batik", "indonesia javanese java textile fabric kain 印尼 sarong"),
        ("rendang", "padang minangkabau beef sapi curry indonesia malaysia 印尼 nasi"),
        ("nasi goreng", "indonesia 印尼 sambal mie satay sate kecap"),
        ("Komodo dragon", "lizard lizards reptile indonesia flores varanus monitor 蜥蜴 印尼"),
        ("Krakatoa", "volcano eruption 1883 indonesia sunda tsunami 火山 java sumatra"),
        ("gamelan", "java javanese bali balinese music gong orchestra indonesia 印尼 音乐"),
        ("wayang kulit", "java javanese dalang gamelan mahabharata ramayana indonesia"),
        ("Pancasila", "indonesia ideology soekarno sukarno garuda 印尼 principles uud"),
        ("Lake Toba", "sumatra batak volcano supervolcano medan samosir indonesia 印尼"),
        ("Raja Ampat", "papua diving coral reef reefs indonesia 印尼 sorong"),
        ("Majapahit", "empire kingdom java javanese hayam wuruk gajah mada hindu kerajaan"),
        ("Garuda Indonesia", "airline airlines flights aviation airport jakarta 航空 maskapai lion"),
        ("Gojek", "grab ojek motorcycle ride startup tokopedia jakarta uber"),
        ("angklung", "bamboo bambu sunda sundanese bandung music instrument gamelan"),
        ("the Dutch East Indies", "batavia colonial colony indonesia voc java 印尼 sumatra"),
        ("Mount Bromo", "volcano java tengger semeru malang crater 火山 indonesia"),
        ("sate ayam", "peanut kacang sauce skewer skewers indonesia madura lontong"),
        ("Toraja", "sulawesi funeral funerals tongkonan makassar indonesia 印尼 buffalo"),
        ("Kartini", "women emancipation raden jepara feminism indonesia 印尼 heroine pahlawan"),
    ],
    "everyday": [
        ("coffee", "caffeine espresso latte cappuccino tea starbucks beans 茶"),
        ("a toothbrush", "toothpaste teeth dental 牙膏 牙齿 floss dentist"),
        ("an umbrella", "rain raining rainy 雨 下雨 wet"),
        ("a bicycle", "cycling pedal pedals wheels cyclist helmet"),
        ("the refrigerator", "freezer cold food ice kitchen 冷冻"),
        ("a wedding", "bride groom marriage married 新娘 结婚 honeymoon"),
        ("breakfast", "eggs cereal morning pancakes toast coffee lunch 早上"),
        ("the dentist", "teeth tooth dental 牙齿 cavity cavities orthodontist"),
        ("a passport", "visa travel airport immigration 签证 customs citizenship"),
        ("homework", "school students student teacher assignment assignments 学生 exam"),
        ("a smartphone", "iphone android apps app samsung apple touchscreen"),
        ("the alarm clock", "wake waking morning snooze sleep 早上"),
        ("a birthday party", "cake candles presents gifts balloons 蛋糕 celebrate"),
        ("rush hour", "traffic commute commuters 交通 congestion jam subway"),
        ("the laundry", "washing detergent clothes dryer 衣服 wash"),
    ],
}
PILOT = [l.strip() for l in open("bench/dev_phrases.txt", encoding="utf-8") if l.strip() and not l.startswith("#")][:30]


def norm(w):
    return unicodedata.normalize("NFKC", w).lower()


def main():
    pilot = {norm(p) for p in PILOT}
    out, seen = [], set()
    for cat, items in P.items():
        for ph, rel in items:
            words = {norm(w) for w in ph.replace("-", " ").split()}
            r = rel.split()
            assert len(r) == len(set(r)), ph
            assert len(r) >= 4, ph
            clash = [w for w in r if norm(w) in words]
            assert not clash, (ph, clash)
            assert norm(ph) not in pilot and norm(ph) not in seen, ph
            seen.add(norm(ph))
            out.append({"id": len(out) + 1, "category": cat, "phrase": ph, "related": r})
    with open("bench/phrases.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(len(out), dict(Counter(o["category"] for o in out)))


if __name__ == "__main__":
    main()
