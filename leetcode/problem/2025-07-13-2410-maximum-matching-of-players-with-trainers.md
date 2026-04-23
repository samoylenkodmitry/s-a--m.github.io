---
layout: leetcode-entry
title: "2410. Maximum Matching of Players With Trainers"
permalink: "/leetcode/problem/2025-07-13-2410-maximum-matching-of-players-with-trainers/"
leetcode_ui: true
entry_slug: "2025-07-13-2410-maximum-matching-of-players-with-trainers"
---

[2410. Maximum Matching of Players With Trainers](https://leetcode.com/problems/maximum-matching-of-players-with-trainers/description/) medium
[blog post](https://leetcode.com/problems/maximum-matching-of-players-with-trainers/solutions/6952394/kotlin-rust-by-samoylenkodmitry-syx0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13072025-2410-maximum-matching-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/C6E7o7iRoI8)
![1.webp](/assets/leetcode_daily_images/b3a08fcb.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1048

#### Problem TLDR

Count trainers for players #medium #sort

#### Intuition

Skip not able trainers, always take smallest player and trainer.

#### Approach

* we can iterate over trainers or over players
* Rust `into_iter` is slower than `iter` 7ms vs 3ms
* Rust `iter` is slower than `for` 3ms vs 0ms
* have to sort both, 10^9 range not suitable for counting sort linear solution
* players counter and result counter is the same variable

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 49ms
    fun matchPlayersAndTrainers(ps: IntArray, ts: IntArray): Int {
        ps.sort(); ts.sort(); var t = 0; var cnt = 0
        return ps.count { p ->
            while (t < ts.size && ts[t] < p) ++t
            t++ < ts.size
        }
    }

```
```kotlin

// 44ms
    fun matchPlayersAndTrainers(ps: IntArray, ts: IntArray): Int {
        ps.sort(); ts.sort(); var p = 0
        for (t in ts) {
            if (t < ps[p]) continue
            if (++p == ps.size) break
        }
        return p
    }

```
```rust

// 3ms
    pub fn match_players_and_trainers(mut ps: Vec<i32>, mut ts: Vec<i32>) -> i32 {
        ps.sort_unstable(); ts.sort_unstable(); let mut t = 0;
        ps.iter().take_while(|&&p| {
            while t < ts.len() && ts[t] < p { t += 1 }
            t += 1; t <= ts.len()
        }).count() as _
    }

```
```rust

// 0ms
    pub fn match_players_and_trainers(mut ps: Vec<i32>, mut ts: Vec<i32>) -> i32 {
        ps.sort_unstable(); ts.sort_unstable(); let mut p = 0;
        for t in ts {
            if t < ps[p] { continue }; p += 1;
            if p == ps.len() { break }
        } p as _
    }

```
```c++

// 32ms
    int matchPlayersAndTrainers(vector<int>& ps, vector<int>& ts) {
        sort(begin(ps), end(ps)); sort(begin(ts), end(ts));
        int p = 0;
        for (int t = 0; t < size(ts) && p < size(ps); ++t) p += ts[t] >= ps[p];
        return p;
    }

```

