---
layout: leetcode-entry
title: "386. Lexicographical Numbers"
permalink: "/leetcode/problem/2025-06-08-386-lexicographical-numbers/"
leetcode_ui: true
entry_slug: "2025-06-08-386-lexicographical-numbers"
---

[386. Lexicographical Numbers](https://leetcode.com/problems/lexicographical-numbers/description/) medium
[blog post](https://leetcode.com/problems/lexicographical-numbers/solutions/6822484/kotlin-rust-by-samoylenkodmitry-n149/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08062025-386-lexicographical-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Uq3A3tyoJKk)
![1.webp](/assets/leetcode_daily_images/d6740b6d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1013

#### Problem TLDR

Generate lexicographical numbers 1..n #medium

#### Intuition

```j
    // 1 10 100 1000 11 12 120 2 20 200 21
```
There is a DFS pattern in the order: `1`, `2`, `3` are the headers, with fillers in-between.

#### Approach

* the iterative variant is clever: go deep by *10, then increment, then remove all zeros
* by rewriting the order, some interesting implementations are possible, like runningFold/scan

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 22ms
    fun lexicalOrder(n: Int) = buildList<Int> {
        var x = 1
        repeat(n) {
            add(x)
            if (x * 10 <= n) x *= 10
            else {
                if (x == n) x /= 10
                x++
                while (x % 10 == 0) x /= 10
            }
        }
    }

```
```kotlin

// 19ms
    fun lexicalOrder(n: Int) = buildList<Int> {
        fun dfs(p: Int) {
            if (p > n) return@dfs
            add(p); for (d in 0..9) dfs(p * 10 + d)
        }
        for (d in 1..9) dfs(d)
    }

```
```kotlin

// 18ms
    fun lexicalOrder(n: Int) =
        (1..<n).runningFold(1) { r, t -> var x = r
            if (x * 10 <= n) x *= 10
            else {
                if (x == n) x /= 10
                x++
                while (x % 10 == 0) x /= 10
            }
            x
        }

```
```kotlin

// 6ms
    fun lexicalOrder(n: Int): List<Int> {
        var x = 0
        return List<Int>(n) {
            if (x > 0 && x * 10 <= n) x *= 10
            else {
                if (x == n) x /= 10
                x++
                while (x % 10 == 0) x /= 10
            }
            x
        }
    }

```

```rust

// 0ms
    pub fn lexical_order(n: i32) -> Vec<i32> {
        (0..n).scan(0, |x, t| {
            if *x > 0 && *x * 10 <= n { *x *= 10 }
            else {
                if *x == n { *x /= 10 }
                *x += 1;
                while *x % 10 == 0 { *x /= 10 }
            }; Some(*x)
        }).collect()
    }

```
```c++

// 1ms
    vector<int> lexicalOrder(int n) {
        int x = 1; vector<int>r(n);
        for (int i = 0; i < n; ++i) {
            r[i] = x;
            if (x * 10 <= n) x *= 10;
            else {
                if (x == n) x /= 10;
                ++x;
                while (x % 10 == 0) x /= 10;
            }
        } return r;
    }

```

