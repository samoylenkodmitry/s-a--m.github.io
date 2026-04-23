---
layout: leetcode-entry
title: "1912. Design Movie Rental System"
permalink: "/leetcode/problem/2025-09-21-1912-design-movie-rental-system/"
leetcode_ui: true
entry_slug: "2025-09-21-1912-design-movie-rental-system"
---

[1912. Design Movie Rental System](https://leetcode.com/problems/design-movie-rental-system/description) medium
[blog post](https://leetcode.com/problems/design-movie-rental-system/solutions/7210372/kotlin-rust-by-samoylenkodmitry-btis/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21092025-1912-design-movie-rental?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3tXT9XTqlRg)

![1.webp](/assets/leetcode_daily_images/06a2b784.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1119

#### Problem TLDR

Design Movie Rent: rent,drop,search(5 lowest), report(5 lowest rented) #hard #ds

#### Intuition

To search by movie use a HashMap movie-TreeSet(price|shop).
To report 5 lowest rented use a TreeSet<(price|shop|movie)>.

#### Approach

* TreeSet uses comparator to check uniqness, add movie to the key

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 608ms
class MovieRentingSystem(n: Int, e: Array<IntArray>):
TreeSet<IntArray>(compareBy({it[2]},{it[0]},{it[1]})) {
    val smp = e.groupBy {it[0]}.mapValues{(_,v) -> v.associate {it[1] to it}}
    val mps = e.groupBy {it[1]}.mapValues{(_,v) -> TreeSet(comparator()).also{it+=v}}
    fun search(m: Int) = mps[m]?.take(5)?.map {it[0]} ?: listOf()
    fun rent(s: Int, m: Int) = smp[s]!![m]!!.let {this += it; mps[m]!! -= it}
    fun drop(s: Int, m: Int) = smp[s]!![m]!!.let {this -= it; mps[m]!! += it}
    fun report() = take(5).map {it.take(2)}
}

```
```rust

// 99ms
type PSM = (i32,i32,i32); #[derive(Default)]
struct MovieRentingSystem(BTreeSet<PSM>,HashMap<i32,HashMap<i32,PSM>>,HashMap<i32,BTreeSet<PSM>>);
impl MovieRentingSystem {
    fn new(_: i32, e: Vec<Vec<i32>>) -> Self {
        let mut s = Self::default(); for v in e { let t=(v[2],v[0],v[1]);
        s.1.entry(v[0]).or_default().insert(v[1],t); s.2.entry(v[1]).or_default().insert(t);}; s }
    fn search(&self, m: i32)->Vec<i32>{self.2.get(&m).iter().flat_map(|s|s.iter().take(5).map(|t|t.1)).collect() }
    fn rent(&mut self, s: i32, m: i32){let t=self.1[&s][&m];self.0.insert(t);self.2.get_mut(&m).unwrap().remove(&t);}
    fn drop(&mut self, s: i32, m: i32){let t=self.1[&s][&m];self.0.remove(&t);self.2.get_mut(&m).unwrap().insert(t);}
    fn report(&self) -> Vec<Vec<i32>> {self.0.iter().take(5).map(|t|vec![t.1,t.2]).collect()}
}

```

