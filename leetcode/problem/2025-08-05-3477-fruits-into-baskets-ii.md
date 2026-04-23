---
layout: leetcode-entry
title: "3477. Fruits Into Baskets II"
permalink: "/leetcode/problem/2025-08-05-3477-fruits-into-baskets-ii/"
leetcode_ui: true
entry_slug: "2025-08-05-3477-fruits-into-baskets-ii"
---

[3477. Fruits Into Baskets II](https://leetcode.com/problems/fruits-into-baskets-ii/description/) easy
[blog post](https://leetcode.com/problems/fruits-into-baskets-ii/solutions/7046456/kotlin-rust-by-samoylenkodmitry-w442/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/5082025-3477-fruits-into-baskets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/WpxdBVT88Zw)
![1.webp](/assets/leetcode_daily_images/a46aff59.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1071

#### Problem TLDR

Place fruits left to right to first available bucket #easy #simulation

#### Intuition

Simulate the process, brute-force is accepted.

#### Approach

* the code is simpler with decreasing result from size
* refresh your memory about segment trees: range 4length, 2i+1,2i+2, compare max(l, r)

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$ or O(n) if not modyfying the inputs

#### Code

```kotlin

// 24ms
    fun numOfUnplacedFruits(f: IntArray, b: IntArray) = f.size -
        f.count { f -> b.indices.any { i -> (b[i] >= f).also { if (it) b[i] = 0}}}

```
```kotlin

// 13ms
    fun numOfUnplacedFruits(f: IntArray, b: IntArray): Int {
        val s = IntArray(4 * b.size)
        fun make(l: Int, h: Int, i: Int): Int {
            if (l == h) { s[i] = b[l]; return s[i] }
            val m = (l + h) / 2
            s[i] = max(make(l, m, 2 * i + 1), make(m + 1, h, 2 * i + 2))
            return s[i]
        }
        fun q(x: Int, l: Int, h: Int, i: Int): Boolean =
            if (l == h) if (s[i] >= x) { s[i] = 0; true } else false
            else if (s[i] >= x) {
                val m = (l + h) / 2
                val r = q(x, l, m, 2 * i + 1) || q(x, m + 1, h, 2 * i + 2)
                s[i] = max(s[2 * i + 1], s[2 * i + 2]); r
            } else false
        make(0, b.lastIndex, 0)
        return f.size - f.count { q(it, 0, b.lastIndex, 0) }
    }

```
```kotlin

// 3ms
    fun numOfUnplacedFruits(f: IntArray, b: IntArray): Int {
        var res = f.size
        for (f in f) for (i in b.indices)
            if (b[i] >= f) { b[i] = 0; res--; break }
        return res
    }

```

```rust

// 0ms
    pub fn num_of_unplaced_fruits(f: Vec<i32>, mut b: Vec<i32>) -> i32 {
        let mut r = f.len() as i32;
        for f in f { for i in 0..b.len() { if b[i] >= f { b[i] = 0; r -= 1; break }}} r
    }

```
```c++

// 0ms
    int numOfUnplacedFruits(vector<int>& f, vector<int>& b) {
        int c = 0;
        for (int f: f) for (int& b: b) if (b >= f) { b = 0, ++c; break; };
        return size(b) - c;
    }

```
```python

// 19ms
    def numOfUnplacedFruits(self, f: List[int], b: List[int]) -> int:
        r = len(b)
        for f in f:
            for i, v in enumerate(b):
                if v >= f: b[i] = 0; r -= 1; break
        return r

```

