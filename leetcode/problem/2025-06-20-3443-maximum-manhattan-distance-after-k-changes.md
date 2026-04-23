---
layout: leetcode-entry
title: "3443. Maximum Manhattan Distance After K Changes"
permalink: "/leetcode/problem/2025-06-20-3443-maximum-manhattan-distance-after-k-changes/"
leetcode_ui: true
entry_slug: "2025-06-20-3443-maximum-manhattan-distance-after-k-changes"
---

[3443. Maximum Manhattan Distance After K Changes](https://leetcode.com/problems/maximum-manhattan-distance-after-k-changes/description) medium
[blog post](https://leetcode.com/problems/maximum-manhattan-distance-after-k-changes/solutions/6864641/kotlin-rust-by-samoylenkodmitry-l5v9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20062025-3443-maximum-manhattan-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DwSEXjLTJJ4)
![1.webp](/assets/leetcode_daily_images/34a10025.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1025

#### Problem TLDR

Max distance after k flips #medium

#### Intuition

Used the hints.

```j

    // find final vector dx dy
    // change k negative parts
    // SWWEW    k=1
    //     .    dy=-1
    //   ...    dx=-2
    // order doesn't matter

    // NWSE        . N E     NWNE
    //             . W N
    //             . . .

    // i solved the wrong problem, the MAX is ON the path, not final
    // the order MATTER

```

* we have to check each step
* at each step remove the opposite to the maximum direction

Another clever intuition is `min(total, dist + 2k)`:
* each flip do +2
* max flips we can do is `total`

#### Approach

* pay attention to the description, don't solve the wrong problem

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 86ms
    fun maxDistance(str: String, k: Int): Int {
        var n = 0; var e = 0; var w = 0; var s = 0
        return str.withIndex().maxOf { (i, c) ->
            when (c) { 'N' -> ++n; 'S' -> ++s; 'W' -> ++w; 'E' -> ++e }
            i + 1 - 2 * max(0, minOf(w + s, e + n, w + n, e + s) - k)
        }
    }

```
```kotlin

// 58ms
    fun maxDistance(str: String, k: Int): Int {
        var res = 0; var n = 0; var e = 0; var w = 0; var s = 0
        for (c in str) {
            when (c) { 'N' -> ++n; 'S' -> ++s; 'W' -> ++w; 'E' -> ++e }
            if (e >= w && n >= s) {
                val bad = max(0, w + s - k)
                val good = w + s - bad
                res = max(res, e + n - bad + good)
            } else if (e < w && n < s) {
                val bad = max(0, e + n - k)
                val good = e + n - bad
                res = max(res, w + s - bad + good)
            } else if (e >= w && n < s) {
                val bad = max(0, w + n - k)
                val good = w + n - bad
                res = max(res, e + s - bad + good)
            } else {
                val bad = max(0, e + s - k)
                val good = e + s - bad
                res = max(res, w + n - bad + good)
            }
        }
        return res
    }

```
```rust

// 90ms
    pub fn max_distance(st: String, k: i32) -> i32 {
        let (mut n, mut e, mut w, mut s) = (0, 0, 0, 0);
        st.bytes().enumerate().map(|(i, c)| {
            match (c) { b'N' => n += 1, b'S' => s += 1, b'E' => e += 1, _ => w += 1 }
            i as i32 + 1 - 2 * 0.max((w + s).min(e + n).min(w + n).min(e + s) - k)
        }).max().unwrap()
    }

```
```rust

// 32ms
    pub fn max_distance(st: String, k: i32) -> i32 {
        let (mut n, mut e, mut w, mut s, mut r) = (0, 0, 0, 0, 0);
        for (i, b) in st.bytes().enumerate() {
            match (b) { b'N' => n += 1, b'S' => s += 1, b'E' => e += 1, _ => w += 1 }
            r = r.max(i as i32 + 1 - 2 * 0.max((w + s).min(e + n).min(w + n).min(e + s) - k))
        } r
    }

```
```rust

// 31ms
    pub fn max_distance(st: String, k: i32) -> i32 {
        let (mut x, mut y, mut r) = (0i32, 0i32, 0);
        for (i, b) in st.bytes().enumerate() {
            match (b) { b'N' => y += 1, b'S' => y -= 1, b'E' => x += 1, _ => x -= 1 }
            r = r.max((i as i32 + 1).min(x.abs() + y.abs() + 2 * k))
        } r
    }

```
```c++

// 22ms
    int maxDistance(string s, int k) {
        int x = 0, y = 0, r = 0, total = 1;
        for (char c: s) {
            y += (c == 'N') - (c == 'S'); x += (c == 'E') - (c == 'W');
            r = max(r, min(total++, abs(x) + abs(y) + 2 * k));
        } return r;
    }

```

