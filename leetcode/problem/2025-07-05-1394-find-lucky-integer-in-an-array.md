---
layout: leetcode-entry
title: "1394. Find Lucky Integer in an Array"
permalink: "/leetcode/problem/2025-07-05-1394-find-lucky-integer-in-an-array/"
leetcode_ui: true
entry_slug: "2025-07-05-1394-find-lucky-integer-in-an-array"
---

[1394. Find Lucky Integer in an Array](https://leetcode.com/problems/find-lucky-integer-in-an-array/description) easy
[blog post](https://leetcode.com/problems/find-lucky-integer-in-an-array/solutions/6922267/kotlin-rust-by-samoylenkodmitry-0tjw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/5072025-1394-find-lucky-integer-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-2pzlHKEU4M)
![1.webp](/assets/leetcode_daily_images/6e42c76f.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1040

#### Problem TLDR

Max x == freq(x) #easy

#### Intuition

The most brute-force is O(n^2), the fastest is O(n) and O(1) memory.

#### Approach

* how many ways to write this code?
* skip all numbers bigger than size
* sort, group, chunk, build a table, bit shift

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 14ms
    fun findLucky(arr: IntArray): Int {
        arr.sortDescending(); var c = 0; var p = -1
        for (x in arr) if (x == p) ++c else { if (c == p) return p; c = 1; p = x }
        return if (c == p) p else -1
    }

```
```kotlin

// 13ms
    fun findLucky(arr: IntArray) =
        (500 downTo 1).firstOrNull { x -> x == arr.count { it == x } } ?: -1

```
```kotlin

// 6ms
    fun findLucky(arr: IntArray) =
        arr.groupBy { it }.maxOf { (k, v) -> if (v.size == k) k else -1 }

```
```kotlin

// 2ms
    fun findLucky(a: IntArray): Int {
        for (x in a) if ((x and 0xfff) <= a.size) a[(x and 0xfff) - 1] += 1 shl 12
        for (x in a.size downTo 1) if (x == a[x - 1] shr 12) return x
        return -1
    }

```
```kotlin

// 1ms
    fun findLucky(arr: IntArray): Int {
        val f = IntArray(501); for (x in arr) ++f[x]
        for (x in arr.size downTo 1) if (x == f[x]) return x
        return -1
    }

```
```rust

// 0ms
    pub fn find_lucky(mut a: Vec<i32>) -> i32 {
        a.sort_unstable(); a.chunk_by(|a, b| a == b)
        .filter(|c| c.len() as i32 == c[0]).map(|c| c[0] as i32).max().unwrap_or(-1)
    }

```
```c++

// 0ms
    int findLucky(vector<int>& a) {
        int f[501]={}, r = -1; f[0] = 1;
        for (int x: a) ++f[x];
        for (int x: f) if (x == f[x]) r = max(r, x);
        return r;
    }

```

