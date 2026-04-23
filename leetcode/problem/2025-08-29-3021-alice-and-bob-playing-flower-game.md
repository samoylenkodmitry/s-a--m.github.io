---
layout: leetcode-entry
title: "3021. Alice and Bob Playing Flower Game"
permalink: "/leetcode/problem/2025-08-29-3021-alice-and-bob-playing-flower-game/"
leetcode_ui: true
entry_slug: "2025-08-29-3021-alice-and-bob-playing-flower-game"
---

[3021. Alice and Bob Playing Flower Game](https://leetcode.com/problems/alice-and-bob-playing-flower-game/description) medium
[blog post](https://leetcode.com/problems/alice-and-bob-playing-flower-game/solutions/7134227/kotlin-rust-by-samoylenkodmitry-zjls/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29082025-3021-alice-and-bob-playing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7gwxMDE-HWA)

![1.webp](/assets/leetcode_daily_images/8a10a2a3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1096

#### Problem TLDR

Pairs x=1..n * y=1..m for Alice to win in take from (x+y) game #medium

#### Intuition

Count odd x+y. It is sum of first odd+second even or reversed.

The interesting math fact from others (can be proved by considering 4 cases: even-even, even-odd,odd-even, odd-odd):
```
n/2 * (m+1)/2 + m/2 * (n+1)/2 == n*m/2
```

#### Approach

* try to write O(1)

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 0ms
    fun flowerGame(n: Int, m: Int) =
        1L * (n/2) * ((m+1)/2) + 1L * ((n+1)/2) * (m/2)

```
```kotlin

// 11ms
    fun flowerGame(n: Int, m: Int): Long {
        val ne = (2..n).count { it % 2 < 1 }
        val me = (2..m).count { it % 2 < 1 }
        return 1L * (n-ne) * me + 1L * ne * (m-me)
    }

```
```rust

// 0ms
    pub fn flower_game(n: i32, m: i32) -> i64 {
        n as i64 * m as i64 / 2
    }

```
```c++

// 0ms
    long long flowerGame(int n, int m) {
        return 1LL * n * m / 2;
    }

```
```python

// 0ms
    flowerGame = lambda _,n,m: n*m//2

```

