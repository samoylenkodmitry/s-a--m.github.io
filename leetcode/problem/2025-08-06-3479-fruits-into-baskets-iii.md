---
layout: leetcode-entry
title: "3479. Fruits Into Baskets III"
permalink: "/leetcode/problem/2025-08-06-3479-fruits-into-baskets-iii/"
leetcode_ui: true
entry_slug: "2025-08-06-3479-fruits-into-baskets-iii"
---

[3479. Fruits Into Baskets III](https://leetcode.com/problems/fruits-into-baskets-iii/description) medium
[blog post](https://leetcode.com/problems/fruits-into-baskets-iii/solutions/7050464/kotlin-rust-by-samoylenkodmitry-shkj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6082025-3479-fruits-into-baskets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/n0MeL66o6NI)
![1.webp](/assets/leetcode_daily_images/ef2eda6e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1072

#### Problem TLDR

Place fruits left to right to first available bucket #medium #segment_tree

#### Intuition

Use a segment tree: range 4length, 2i+1,2i+2, compare max(l, r)

#### Approach

* try to memorize how segment tree code looks like
* the iterative segment tree: size of 2*next_power_of_two, copy array to the second part

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 45ms
    fun numOfUnplacedFruits(f: IntArray, b: IntArray): Int {
        var sz = 1; var i = 1; while (sz < b.size) sz *= 2
        val s = IntArray(sz * 2); for (i in b.indices) s[sz + i] = b[i]
        for (i in sz - 1 downTo 1) s[i] = max(s[i*2], s[i*2+1])
        return f.count { f ->
            while (i < sz) { i *= 2; if (s[i] < f) i++ }
            (i >= s.size || s[i] < f).also { if (!it) {
                s[i] = 0; while (i > 1) { s[i/2] = max(s[i], s[i xor 1]); i /= 2 }
            } else i = 1 }
        }
    }

```
```kotlin

// 58ms
    fun numOfUnplacedFruits(f: IntArray, b: IntArray): Int {
        val s = IntArray(b.size * 4)
        fun make(i: Int, l: Int, h: Int): Int {
            if (l == h) { s[i] = b[l]; return s[i] }
            val m = (l + h) / 2
            s[i] = max(make(2*i+1, l, m), make(2*i+2, m + 1, h))
            return s[i]
        }
        fun check(v: Int, i: Int, l: Int, h: Int): Boolean =
            if (l == h) {
                if (s[i] >= v) { s[i] = 0; true } else false
            } else  if (s[i] < v) false else {
                val m = (l + h) / 2
                val r = check(v, 2*i+1, l, m) || check (v, 2*i+2, m + 1, h)
                s[i] = max(s[2*i+1], s[2*i+2]); r
            }
        make(0, 0, b.lastIndex)
        return f.size - f.count { check(it, 0, 0, b.lastIndex) }
    }

```
```rust

// 40ms
    pub fn num_of_unplaced_fruits(f: Vec<i32>, b: Vec<i32>) -> i32 {
        fn make(i: usize, l: usize, h: usize, s: &mut Vec<i32>, b: &Vec<i32>) {
            if l == h { s[i] = b[l] } else {
                let m = (l + h) / 2; make(2*i+1, l, m, s, b); make(2*i+2, m+1, h, s, b);
                s[i] = s[2*i+1].max(s[2*i+2])
            }}
        fn c(v: i32, i: usize, l: usize, h: usize, s: &mut Vec<i32>) -> bool {
            if l == h {
                if s[i] >= v { s[i] = 0; true } else { false }
            } else if s[i] < v { false } else {
                let m = (l + h) / 2; let r = c(v, 2*i+1, l, m, s) || c(v, 2*i+2, m+1, h, s);
                s[i] = s[2*i+1].max(s[2*i+2]); r
            }}
        let mut s = vec![0; b.len() * 4]; make(0, 0, b.len()-1, &mut s, &b);
        (f.len() - (0..f.len()).filter(|&i| c(f[i], 0, 0, b.len()-1, &mut s)).count()) as _
    }

```
```rust

// 27ms
    pub fn num_of_unplaced_fruits(f: Vec<i32>, b: Vec<i32>) -> i32 {
        let sz = b.len().next_power_of_two(); let mut s = vec![0; 2 * sz];
        s[sz..sz + b.len()].copy_from_slice(&b); let mut r = 0;
        for i in (1..sz).rev() { s[i] = s[2*i].max(s[2*i+1]) }
        for f in f {
            let mut i = 1; while i < sz { i *= 2; if s[i] < f { i += 1 }}
            if i >= sz && s[i] >= f {
                s[i] = 0; while i > 1 { i /= 2; s[i] = s[i*2].max(s[i*2+1]) }
            } else { r += 1 }
        } r
    }

```
```c++

// 45ms
    int numOfUnplacedFruits(vector<int>& f, vector<int>& b) {
        int n = size(b), sz = 1, r = 0; while (sz < n) sz <<= 1;
        vector<int> s(2*sz); copy(begin(b), end(b), begin(s) + sz);
        for (int i = sz-1; i; --i) s[i] = max(s[i<<1], s[i<<1|1]);
        for (int f: f) {
            int i = 1; while (i < sz) { i <<= 1; if (s[i] < f) ++i; }
            if (i < 2*sz && s[i] >= f) {
                s[i] = 0; for (; i; i >>= 1) s[i>>1] = max(s[i], s[i^1]);
            } else ++r;
        } return r;
    }

```
```python

// 1591ms
    def numOfUnplacedFruits(self, f: List[int], b: List[int]) -> int:
        n = len(b); sz = 1
        while sz < n: sz *= 2
        s = [0] * (2 * sz); s[sz:sz + n] = b; r = 0
        for i in range(sz - 1, 0, -1): s[i] = max(s[i<<1], s[i<<1|1])
        for f in f:
            i = 1
            while i < sz: i <<= 1; i += s[i] < f
            if i < 2 * sz and s[i] >= f:
                s[i] = 0;
                while i: s[i>>1] = max(s[i], s[i^1]); i >>= 1
            else: r += 1
        return r

```

