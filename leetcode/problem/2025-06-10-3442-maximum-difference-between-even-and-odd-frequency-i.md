---
layout: leetcode-entry
title: "3442. Maximum Difference Between Even and Odd Frequency I"
permalink: "/leetcode/problem/2025-06-10-3442-maximum-difference-between-even-and-odd-frequency-i/"
leetcode_ui: true
entry_slug: "2025-06-10-3442-maximum-difference-between-even-and-odd-frequency-i"
---

[3442. Maximum Difference Between Even and Odd Frequency I](https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-i/description/) easy
[blog post](https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-i/solutions/6828740/kotlin-rust-by-samoylenkodmitry-i31k/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10062025-3442-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Vb8kN0sU21c)
![1.webp](/assets/leetcode_daily_images/77de57d4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1015

#### Problem TLDR

Max odd - min even frequency #easy

#### Intuition

Find all frequencies, then do the search.

#### Approach

* use built-in methods like `groupBy`
* compare with hand-crafted iterative code performance

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 24ms
    fun maxDifference(s: String) = s.groupingBy { it }
      .eachCount().values.groupBy { it % 2 }
      .let { it[1]!!.max() - it[0]!!.min() }

```
```kotlin

// 23ms
    fun maxDifference(s: String) = s.groupingBy { it }
      .eachCount().values.partition { it % 2 > 0 }
      .let { (a, b) -> a.max() - b.min() }

```
```kotlin

// 8ms
    fun maxDifference(s: String) = s.groupBy { it }.values
      .groupBy { it.size % 2 }
      .let { it[1]!!.maxOf { it.size } - it[0]!!.minOf { it.size } }

```
```kotlin

// 7ms
    fun maxDifference(s: String) = with(s.groupBy { it }.values) {
        filter { it.size % 2 > 0 }.maxOf { it.size } -
        filter { it.size % 2 < 1 }.minOf { it.size }
    }

```
```kotlin

// 6ms
    fun maxDifference(s: String) = with(s.groupBy { it }.values) {
        maxOf { (it.size % 2) * it.size } -
        minOf { it.size + (it.size % 2) * (99 - it.size) }
    }

```
```kotlin

// 1ms
    fun maxDifference(s: String): Int {
        val f = IntArray(26); for (c in s) ++f[c - 'a']
        var a = 0; var b = s.length
        for (f in f) if (f > 0)
            if (f % 2 > 0) a = max(f, a) else b = min(f, b)
        return a - b
    }

```
```rust

// 0ms
    pub fn max_difference(mut s: String) -> i32 {
        let (mut a, mut b, mut f) = (0, 99, [0; 26]);
        for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        for f in f { if (f > 0 && f % 2 < 1) { b = b.min(f) } else { a = a.max(f) }}
        a - b
    }

```
```rust

// 0ms
    pub fn max_difference(mut s: String) -> i32 {
        let mut s = unsafe { s.as_bytes_mut() }; s.sort_unstable();
        let (a, b): (Vec<_>, Vec<_>) = s[..].chunk_by(|a, b| a == b)
        .map(|c| c.len()).partition(|l| l % 2 > 0);
        (a.iter().max().unwrap() - b.iter().min().unwrap()) as _
    }

```
```c++

// 0ms
    int maxDifference(string s) {
        int f[26]={}, a = 0, b = 99;
        for (auto c: s) ++f[c - 'a'];
        for (int c: f) if (c > 0 && !(c & 1)) b = min(b, c); else a = max(a, c);
        return a - b;
    }

```

