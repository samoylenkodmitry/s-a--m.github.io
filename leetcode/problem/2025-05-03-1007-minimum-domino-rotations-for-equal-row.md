---
layout: leetcode-entry
title: "1007. Minimum Domino Rotations For Equal Row"
permalink: "/leetcode/problem/2025-05-03-1007-minimum-domino-rotations-for-equal-row/"
leetcode_ui: true
entry_slug: "2025-05-03-1007-minimum-domino-rotations-for-equal-row"
---

[1007. Minimum Domino Rotations For Equal Row](https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/description/) medium
[blog post](https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/solutions/6710134/kotlin-rust-by-samoylenkodmitry-akgg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03052025-1007-minimum-domino-rotations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BQZUEjwbHJ8)
![1.webp](/assets/leetcode_daily_images/7d817b10.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/977

#### Problem TLDR

Min swaps to make a top or bottom row #medium

#### Intuition

It's all about the implementation and edge cases. Consider counting the frequencies for the top and for the bottom row. Then detect a dominant value. Then check it can fill the row.

#### Approach

* first domino contains a dominant value
* consider both values as possible dominant
* we can use a recursion, or a single iteration with four counters (can be less?)
* in jvm test machine two passes are faster than a single pass, possible beacuse of the instructions prediction or cache can't handle too many variables at once

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 19ms
    fun minDominoRotations(tops: IntArray, bottoms: IntArray, d: Int = 0): Int =
    if (d < 1) {
        val a = minDominoRotations(tops, bottoms, tops[0])
        if (a < 0) minDominoRotations(tops, bottoms, bottoms[0]) else a
    } else if (tops.indices.any { tops[it] != d && bottoms[it] != d }) -1
    else tops.size - max(tops.count { it == d }, bottoms.count { it == d })

```
```kotlin

// 6ms
    fun minDominoRotations(top: IntArray, bottom: IntArray): Int {
        var d1 = top[0]; var d2 = bottom[0]
        var a = top.size; var b = a; var c = a; var d = a
        for (i in 0..<a) {
            val tp = top[i]; val bt = bottom[i]
            if (tp != d1 && bt != d1) if (d2 == 0) return -1 else d1 = 0
            if (tp != d2 && bt != d2) if (d1 == 0) return -1 else d2 = 0
            if (tp == d1) --a else if (tp == d2) --b
            if (bt == d1) --c else if (bt == d2) --d
        }
        return min(min(a, b), min(c, d))
    }

```
```kotlin

// 4ms
    fun minDominoRotations(top: IntArray, bottom: IntArray): Int {
        var d = top[0]; var a = 0; var b = 0
        for (i in top.indices) {
            if (top[i] != d && bottom[i] != d) { a  = -1; b = -1; break }
            if (top[i] == d) ++a; if (bottom[i] == d) ++b
        }
        var r = max(a, b); if (r >= 0) return top.size - r
        d = bottom[0]; a = 0; b = 0
        for (i in top.indices) {
            if (top[i] != d && bottom[i] != d) { a  = -1; b = -1; break }
            if (top[i] == d) ++a; if (bottom[i] == d) ++b
        }
        r = max(a, b)
        return if (r < 0) -1 else top.size - r
    }

```
```rust

// 0ms
    pub fn min_domino_rotations(tops: Vec<i32>, bottoms: Vec<i32>) -> i32 {
        let (mut d1, mut d2, mut a) = (tops[0], bottoms[0], tops.len());
        let (mut b, mut c, mut d) = (a, a, a);
        for i in 0..a {
            let (tp, bt) = (tops[i], bottoms[i]);
            if tp != d1 && bt != d1 { if d2 < 1 { return -1 }; d1 = 0 }
            if tp != d2 && bt != d2 { if d1 < 1 { return -1 }; d2 = 0 }
            if tp == d1 { a -= 1 } else if tp == d2 { b -= 1 }
            if bt == d1 { c -= 1 } else if bt == d2 { d -= 1 }
        } a.min(b).min(c).min(d) as _
    }

```
```c++

// 0ms
    int minDominoRotations(vector<int>& tops, vector<int>& bottoms) {
        int d1 = tops[0], d2 = bottoms[0], j = 0, a = size(tops);
        int b = a, c = a, d = a;
        for (int tp: tops) {
            int bt = bottoms[j++];
            if (tp != d1 && bt != d1) if (!d2) return -1; else d1 = 0;
            if (tp != d2 && bt != d2) if (!d1) return -1; else d2 = 0;
            a -= tp == d1; b -= tp == d2; c -= bt == d1; d -= bt == d2;
        } return min(min(a, b), min(c, d));
    }

```

