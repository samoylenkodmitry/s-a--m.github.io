---
layout: leetcode-entry
title: "1865. Finding Pairs With a Certain Sum"
permalink: "/leetcode/problem/2025-07-06-1865-finding-pairs-with-a-certain-sum/"
leetcode_ui: true
entry_slug: "2025-07-06-1865-finding-pairs-with-a-certain-sum"
---

[1865. Finding Pairs With a Certain Sum](https://leetcode.com/problems/finding-pairs-with-a-certain-sum/description/) medium
[blog post](https://leetcode.com/problems/finding-pairs-with-a-certain-sum/solutions/6926886/kotlin-rust-by-samoylenkodmitry-mmxf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6072025-1865-finding-pairs-with-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kXWn1y2zNsQ)
![1.webp](/assets/leetcode_daily_images/69940d24.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1041

#### Problem TLDR

Design pairs counter; two lists, one changes #medium #hashmap

#### Intuition

The brute-force is accepted: maintain two frequencies map, iterate over the first.

#### Approach

* sort first and exit early
* remove 0 frequency elements

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 231ms
class FindSumPairs(vararg val n: IntArray): HashMap<Int, Int>() {
    init { for (x in n[1]) merge(x, 1, Int::plus) }
    fun add(i: Int, v: Int) {
        merge(n[1][i].also { n[1][i] += v }, -1, Int::plus)
        merge(n[1][i], 1, Int::plus)
    }
    fun count(t: Int) = n[0].sumOf { get(t - it) ?: 0 }
}

```
```kotlin

// 186ms
class FindSumPairs(vararg val n: IntArray): HashMap<Int, Int>() {
    val f1 by lazy {
        val f1 = HashMap<Int, Int>()
        for (x in n[0]) f1.merge(x, 1, Int::plus)
        val af = Array(f1.size) { IntArray(2) }
        var i = 0
        for ((a, f) in f1) { af[i][0] = a; af[i++][1] = f }
        Arrays.sort(af, compareBy { it[0] })
        af
    }
    init { for (x in n[1]) merge(x, 1, Int::plus) }
    fun add(i: Int, v: Int) {
        merge(n[1][i].also { n[1][i] += v }, -1, Int::plus)
        merge(n[1][i], 1, Int::plus)
    }
    fun count(t: Int): Int {
        var r = 0
        for ((a, f) in f1)
            if (a > t) break
            else r += f * (get(t - a) ?: 0)
        return r
    }
}

```
```rust

// 41ms
#[derive(Default)] struct FindSumPairs(Vec<i32>, HashMap<i32, i32>, Vec<i32>);
impl FindSumPairs {
    fn new(mut n1: Vec<i32>, n2: Vec<i32>) -> Self {
        n1.sort_unstable(); let mut m = HashMap::new();
        for &x in &n2 { *m.entry(x).or_insert(0) += 1 }
        Self(n1, m, n2)
    }
    fn add(&mut self, i: i32, v: i32) {
        let x = self.2[i as usize]; self.2[i as usize] = x + v;
        *self.1.entry(x).or_insert(0) -= 1;
        *self.1.entry(x + v).or_insert(0) += 1
    }
    fn count(&self, t: i32) -> i32 {
        self.0.iter().map(|x| self.1.get(&(t - x)).unwrap_or(&0)).sum()
    }
}

```
```rust

// 19ms https://leetcode.com/problems/finding-pairs-with-a-certain-sum/submissions/1688214588
#[derive(Default)] struct FindSumPairs(Vec<(i32, i32)>, HashMap<i32, i32>, Vec<i32>);
impl FindSumPairs {
    fn new(mut n1: Vec<i32>, n2: Vec<i32>) -> Self {
        n1.sort_unstable(); let mut m = HashMap::new();
        for &x in &n2 { *m.entry(x).or_insert(0) += 1 }
        Self(n1.chunk_by(|a, b| a == b).map(|c| (c[0], c.len() as i32)).collect(), m, n2)
    }
    fn add(&mut self, i: i32, v: i32) {
        let x = self.2[i as usize]; self.2[i as usize] = x + v;
        *self.1.entry(x).or_insert(0) -= 1;
        if self.1[&x] == 0 { self.1.remove(&x); }
        *self.1.entry(x + v).or_insert(0) += 1
    }
    fn count(&self, t: i32) -> i32 {
        let mut r = 0;
        for &(x, c) in &self.0 {
            if x > t { break }
            r += c * self.1.get(&(t - x)).unwrap_or(&0)
        } r
    }
}

```
```c++

// 147ms
class FindSumPairs {
public:
    vector<int> n1, n2; unordered_map<int, int> m;
    FindSumPairs(vector<int>& ns1, vector<int>& ns2) {
        n1 = ns1; n2 = ns2; sort(begin(n1), end(n1));
        for (int x: n2) ++m[x];
    }
    void add(int i, int v) { --m[n2[i]]; n2[i] += v; ++m[n2[i]]; }
    int count(int t) {
        int r = 0;
        for (int x: n1) if (x > t) break; else r += m[t - x];
        return r;
    }
};

```
```c++

// 49ms
class FindSumPairs {
public:
    vector<int> n1, n2; unordered_map<int, int> m;
    FindSumPairs(vector<int>& ns1, vector<int>& ns2) {
        swap(n1, ns1); swap(n2, ns2); sort(begin(n1), end(n1));
        for (int x: n2) ++m[x];
    }
    void add(int i, int v) { --m[n2[i]]; n2[i] += v; ++m[n2[i]]; }
    int count(int t) {
        int r = 0;
        for (int x: n1) if (x > t) break; else {
            auto it = m.find(t - x);
            if (it != end(m)) r += it->second;
        } return r;
    }
};

```

