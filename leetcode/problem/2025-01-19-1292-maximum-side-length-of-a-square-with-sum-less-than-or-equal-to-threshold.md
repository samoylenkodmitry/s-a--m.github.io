---
layout: leetcode-entry
title: "1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold"
permalink: "/leetcode/problem/2025-01-19-1292-maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/"
leetcode_ui: true
entry_slug: "2025-01-19-1292-maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold"
---

[1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold](https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/description) medium
[blog post](https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/solutions/7506726/kotlin-rust-by-samoylenkodmitry-ymlt/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19012025-1292-maximum-side-length?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SKYBGRtLiGc)

![4395fb37-c24d-497d-b519-7265fd6064bb (1).webp](/assets/leetcode_daily_images/67838260.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1242

#### Problem TLDR

Max square with sum less than threshold #medium #prefix-sum

#### Intuition

```j
    // let's try with brute-force 300^4, its 10^6
    // ok, this is TLE
    // we have to do prefix sums...
```

Brute-force is not accepted.
Optimize with prefix sums of all rectangles with top left corner 0,0.

#### Approach

* optimization: binary search the size
* optimization: increase the size by checking size+1 is a new max
* we only have to go +1 to the top-left, because the previous is at most S (it is not possible to "discover" bigger than s+1, otherwise we would already have the more than 's')

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$, can be in-place

#### Code

```kotlin
// 9ms
    fun maxSideLength(m: Array<IntArray>, t: Int): Int {
        var s = 0; val d = Array(301) { IntArray(301) }
        for (y in m.indices) for (x in m[0].indices) {
            d[y + 1][x + 1] = d[y][x + 1] + d[y + 1][x] - d[y][x] + m[y][x]
            if (y>=s && x>=s && t>=d[y+1][x+1]-d[y-s][x+1]-d[y+1][x-s]+d[y-s][x-s])++s
        }
        return s
    }
```
```rust
// 0ms
    pub fn max_side_length(m: Vec<Vec<i32>>, t: i32) -> i32 {
        let (mut d, mut s) = ([[0; 301]; 301], 0);
        for (i, j) in iproduct!(0..m.len(), 0..m[0].len()) { if {
            d[i+1][j+1] = d[i][j+1] + d[i+1][j] - d[i][j] + m[i][j];
            i>=s && j>=s && t>=d[i+1][j+1]-d[i-s][j+1]-d[i+1][j-s]+d[i-s][j-s]}
        {s+=1}} s as i32
    }
```

