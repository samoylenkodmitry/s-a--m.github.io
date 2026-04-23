---
layout: leetcode-entry
title: "1550. Three Consecutive Odds"
permalink: "/leetcode/problem/2025-05-11-1550-three-consecutive-odds/"
leetcode_ui: true
entry_slug: "2025-05-11-1550-three-consecutive-odds"
---

[1550. Three Consecutive Odds](https://leetcode.com/problems/three-consecutive-odds/description/) easy
[blog post](https://leetcode.com/problems/three-consecutive-odds/solutions/6733247/kotlin-rust-by-samoylenkodmitry-becf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11052025-1550-three-consecutive-odds?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zlC_SDO8Jus)
![1.webp](/assets/leetcode_daily_images/ca8178b4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/985

#### Problem TLDR

3 odds #easy #bitmask

#### Intuition

Count odds.

#### Approach

* use bit `& 1` to check for odds
* use bitmask `0b111 = 7` to check for 3 odds

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin []

// 22ms
    fun threeConsecutiveOdds(a: IntArray) =
        "1, 1, 1" in "" + a.map { it % 2 }

```
```kotlin

// 21ms
    fun threeConsecutiveOdds(a: IntArray) =
        a.asList().windowed(3).any { it.all { it % 2 > 0 }}

```
```kotlin

// 23ms
    fun threeConsecutiveOdds(a: IntArray) =
        a.asList().windowed(3).any { it.reduce(Int::and) % 2 > 0 }

```
```kotlin

// 4ms
    fun threeConsecutiveOdds(a: IntArray) = (1..<a.size - 1)
        .any { a[it - 1] and a[it] and a[it + 1] % 2 > 0 }

```
```kotlin

// 0ms https://leetcode.com/problems/three-consecutive-odds/submissions/1630809266
    fun threeConsecutiveOdds(a: IntArray): Boolean {
        var c = 0
        return a.any { c = (it % 2) * (c + 1); c > 2 }
    }

```
```kotlin

// 0ms
    fun threeConsecutiveOdds(a: IntArray): Boolean {
        var c = 0
        return a.any { c = it and 1 or (c shl 1) and 7; c > 6 }
    }

```
```rust

// 0ms https://leetcode.com/problems/three-consecutive-odds/submissions/1630796680
    pub fn three_consecutive_odds(a: Vec<i32>) -> bool {
        a[..].windows(3).any(|w| 0 < 1 & w[0] & w[1] & w[2])
    }

```
```c++

// 0ms
    bool threeConsecutiveOdds(vector<int>& a) {
        for(int c = 0; int &x: a) if ((c = x & 1 | (c << 1) & 7) > 6)
        return 1; return 0;
    }

```

