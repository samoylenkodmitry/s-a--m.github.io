---
layout: leetcode-entry
title: "1695. Maximum Erasure Value"
permalink: "/leetcode/problem/2025-07-22-1695-maximum-erasure-value/"
leetcode_ui: true
entry_slug: "2025-07-22-1695-maximum-erasure-value"
---

[1695. Maximum Erasure Value](https://leetcode.com/problems/maximum-erasure-value/description/) medium
[blog post](https://leetcode.com/problems/maximum-erasure-value/solutions/6988920/kotlin-rust-by-samoylenkodmitry-f80h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22072025-1695-maximum-erasure-value?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/takSLm2JNZA)
![1.webp](/assets/leetcode_daily_images/11469a3b.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1057

#### Problem TLDR

Max unique subarray sum #medium #sliding_window

#### Intuition

Sliding window:
* expand every time
* shrink until condition

#### Approach

* use array for frequency map
* current index is irrelevant, just decrease the frequency

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 33ms
    fun maximumUniqueSubarray(n: IntArray): Int {
        val f = IntArray(10001); var j = 0; var s = 0
        return n.maxOf { x -> ++f[x]; s += x
            while (f[x] > 1) { s -= n[j]; --f[n[j++]] }; s
        }
    }

```
```kotlin

// 8ms
    fun maximumUniqueSubarray(n: IntArray): Int {
        val f = IntArray(10001); var j = 0; var r = 0; var s = 0
        for (x in n) {
            ++f[x]; s += x
            while (f[x] > 1) { s -= n[j]; --f[n[j++]] }
            r = max(r, s)
        }
        return r
    }

```
```rust

// 0ms
    pub fn maximum_unique_subarray(n: Vec<i32>) -> i32 {
        let (mut f, mut j, mut s) = ([0; 10001], 0, 0);
        n.iter().map(|&x| {
            f[x as usize] += 1; s += x;
            while f[x as usize] > 1 { s -= n[j]; f[n[j] as usize ] -= 1; j += 1 }; s
        }).max().unwrap()
    }

```
```c++

// 3ms
    int maximumUniqueSubarray(vector<int>& n) {
        int f[10001] = {}, j = 0, r = 0, s = 0;
        for (auto x: n) {
            ++f[x]; s += x;
            while (f[x] > 1) s -= n[j], --f[n[j++]];
            r = max(r, s);
        } return r;
    }

```

