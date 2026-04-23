---
layout: leetcode-entry
title: "3085. Minimum Deletions to Make String K-Special"
permalink: "/leetcode/problem/2025-06-21-3085-minimum-deletions-to-make-string-k-special/"
leetcode_ui: true
entry_slug: "2025-06-21-3085-minimum-deletions-to-make-string-k-special"
---

[3085. Minimum Deletions to Make String K-Special](https://leetcode.com/problems/minimum-deletions-to-make-string-k-special/description/) medium
[blog post](https://leetcode.com/problems/minimum-deletions-to-make-string-k-special/solutions/6868044/kotlin-rust-by-samoylenkodmitry-44od/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21062025-3085-minimum-deletions-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/me6Ij1p8qEc)
![1.webp](/assets/leetcode_daily_images/fb58da10.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1026

#### Problem TLDR

Min remove to make frequencies k-narrow #medium

#### Intuition

(spent too much time thinking about optimizing 26^2 brute force to 26log(26))
```j
    // calc freq
    // 13579    k=4
    //   *** remove 1 and 3, 4 chars
    // ***55 remove 2 and 4, 6 chars

    // aabcaba
    // a=4 b=2 c=1
    // 1 2 4       k=0
    // * 1 1    remove 1+3=4
    //   * 2    remove 1+2=3
    //     *    remove 1+2=3
    // optimal way to peek the baseline?
    // prefix sum
    // 1 2 4
    //     4
    //   6
    // 7
    // baseline:
    // 1   total_right = 6, new_total_right = 2 * 1 = 2, diff=6-2=4
    //   2 left `1` is removed, right=4, new_right=1*2=2, diff=4-2=2,
    //     4 right=0, left sum=3 is removed
    // actually we only have 26 chars, can brute-force
```

* only the frequencies matters

#### Approach

* pay attention to the intermediate gathered `data size`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 19ms
    fun minimumDeletions(w: String, k: Int): Int {
        val f = IntArray(26); for (c in w) ++f[c - 'a']
        return f.minOf { x -> f.sumOf { if (it < x) it else max(0, it - x - k) }}
    }

```
```kotlin

// 5ms
    fun minimumDeletions(w: String, k: Int): Int {
        val f = IntArray(26); for (c in w) ++f[c - 'a']
        var res = w.length
        for (x in f) if (x > 0) {
            var remove = 0
            for (f in f) if (f > 0) remove += if (f < x) f else max(0, f - x - k)
            res = min(res, remove)
        }
        return res
    }

```
```rust

// 0ms
    pub fn minimum_deletions(w: String, k: i32) -> i32 {
        let mut f = [0; 26]; for c in w.bytes() { f[(c - b'a') as usize] += 1 }
        (0..26).map(|x| (0..26).map(|c| if f[c] < f[x] { f[c] } else { 0.max(f[c] - f[x] - k)}).sum()).min().unwrap()
    }

```

```c++

// 6ms
    int minimumDeletions(string w, int k) {
        int f[26]={}, r = size(w); for (auto& c: w) ++f[c - 'a'];
        for (int x: f) {
            int remove = 0;
            for (int c: f) remove += c < x ? c: max(0, c - x - k);
            r = min(r, remove);
        } return r;
    }

```

