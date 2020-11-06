/*
 * 型は自動推定される。
 * cargoコマンドでプロジェクト管理できる。
 */

/* 型は引数の後ろに書く。*/
fn fun_0(v: &i32) {
    /* 末尾に!が付く関数はマクロである。*/
    println!("{}", v);
}

/* template */
fn print_typename<T>(_: T) {
    println!("{}", std::any::type_name::<T>());
}

/* 構造体 */
struct Person {
    name: String,
    age: i32,
}

/* メソッドの定義 */
impl Person {
    fn get_name(&self) -> &String {
        &self.name /* Pythonライクなアクセス */
    }

    fn get_age(&self) -> &i32 {
        &self.age /* セミコロン書かないとreturnされる。*/
    }

    fn print() {
        println!("hello");
    }
}

pub trait Cry {
    fn cry(&self) {
        /* デフォルトの振る舞い*/
        println!("鳴く");
    }
}

pub struct Cat;
pub struct Dog;
pub struct Bird;

impl Cry for Cat {
    fn cry(&self) {
        println!("ニャーと鳴く");
    }
}

impl Cry for Dog {
    fn cry(&self) {
        println!("ワンと鳴く");
    }
}

/* lifetime parameterと呼ばれる。*/
/* 2つの引数と１つの返り値のlifetime（生存期間）は同じであるとをコンパイラーに教える。*/
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

/* C++のconceptに相当する。*/
/* 引数に取れるオブジェクトはcryメソッドを持たなければならない。*/
fn talk<T>(t: &T)
where
    T: Cry, /* Cry traitsを実装した型に制限 */
{
    t.cry();
}

fn main() {

    //_/_/_/ 型は自動推定
    
    let x = 3; // i32
    let y = String::from("hoge"); // String

    //_/_/_/ 変数はデフォルトでimmutableである。

    //let x = 3;
    //x = 4; /* コンパイル時にエラー! */
    /* 明示的にmutableにする必要がある。*/
    let mut x = 3;
    x = 4;

    //_/_/_/ パターンマッチがある。

    let x = 3;
    match x {
        3 => println!("hello, world!"), /* 各種パターンの記述。アームと呼ばれる。*/
        _ => (),
    }

    //_/_/_/ 呼び出し側で区別ができる参照呼び出し。

    let x = 32;
    fun_0(&x);

    /* 参照はポインタ */
    let y = &x;
    let z: i32 = *y; /* C/C++のように参照外し */

    //_/_/_/ 文字列型
    /* 調べろ。文字列スライスとか */

    let s = "hello"; /* 書き換え不可。いわゆる文字列リテラルのこと。*/
    let t = String::from(s);
    print_typename(s); /* &str 言語に組み込まれた型 */
    print_typename(t); /* String 標準ライブラリにある型 */

    //_/_/_/ 所有権（ownership）は共有できない。
    /* スマートポインタの件も */

    let x = String::from("world");
    let y = x; /* 所有権がyに移る。*/
    //println!("{}", x); /* コンパイル時エラー */
    /* 所有権を失ったオブジェクトに対する操作はコンパイル時に検出される。*/

    //_/_/_/ C++のクラスに相当するものはない。構造体はある。

    let p = Person {
        name: String::from("Kumada"),
        age: 23,
    };

    //_/_/_/ クラスメソッドに相当するものは別途定義する。
    /* コンストラクタも*/
    println!("{} {}", p.get_name(), p.get_age());

    //_/_/_/ HaskellのMaybeに相当するものがある。

    let x = Some(5); /* 型はOption<i32> */
    let y: Option<i32> = None;
    /* パターンマッチで値を取り出す。*/
    match x {
        Some(v) => println!("{}", v),
        None => (),
    }
    match y {
        Some(v) => println!("{}", v),
        None => (),
    }

    //_/_/_/ C++より自由度の高いenumがある。

    enum Message {
        Quit,
        Move { x: i32, y: i32 },    /* struct */
        Write(String),              /* tuple struct */
        ChangeColor(i32, i32, i32), /* tuple struct */
    }

    let x = Message::Quit;
    let y = Message::ChangeColor(255, 255, 255);
    let z = Message::Move { x: 100, y: 200 };
    /* x,y,zの型は全てMessage */
    /* 実はOption<T>はenumである。*/

    //_/_/_/ staticメソッドを定義できる。

    Person::print();

    //_/_/_/ templateがある。

    let v = vec![1, 2, 3];
    /* コンパイル時に置き換えられる真の意味のテンプレート */
    print_typename(v);
    /* 特殊化は? */

    //_/_/_/ traitsがある。

    /* 最近は継承は好まれなくなっている。*/
    /* C++にtraitsに相当するものはない。Javaのinterfaceに相当する機能である。*/
    /* Rustはクラスメソッドを持たないが、traitsを用いて振る舞いを「継承」することができる。*/
    let dog = Dog {};
    let cat = Cat {};
    dog.cry();
    cat.cry();
    talk(&dog);
    talk(&cat);
    let bird = Bird {};
    /* talk(&bird); エラーになる。*/

    //_/_/_/ ポリモーフィズムができる。

    /* DogオブジェクトとCatオブジェクトをヒープ領域に確保する処理。*/
    /* どちらのオブジェクトもCry traitsを実装している。*/
    /* C++と同じ感覚 */
    let mut animals: Vec<Box<dyn Cry>> = vec![Box::new(Dog {}), Box::new(Cat {})];
    for animal in animals {
        animal.cry();
    }

    //_/_/_/ ライフタイムがある。

    let x = String::from("hello");
    let y = String::from("cat");
    let z = longest(&x, &y);
    println!("{}", z);

    //_/_/_/ ユニットテストできる。
}
