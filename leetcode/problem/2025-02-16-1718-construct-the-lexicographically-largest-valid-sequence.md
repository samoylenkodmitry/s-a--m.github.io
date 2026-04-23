---
layout: leetcode-entry
title: "1718. Construct the Lexicographically Largest Valid Sequence"
permalink: "/leetcode/problem/2025-02-16-1718-construct-the-lexicographically-largest-valid-sequence/"
leetcode_ui: true
entry_slug: "2025-02-16-1718-construct-the-lexicographically-largest-valid-sequence"
---

[1718. Construct the Lexicographically Largest Valid Sequence](https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/description/) medium
[blog post](https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/solutions/6428873/kotlin-rust-by-samoylenkodmitry-4g00/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16022025-1718-construct-the-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/a4NTwqLH5W4)
![1.webp](/assets/leetcode_daily_images/f1039673.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/897

#### Problem TLDR

Construct array x = 1..n, a[i] = x, a[i + x] = x #medium #backtracking

#### Intuition

The problem size is 20 elements max, brute-force backtracking works. An example:

```j

    // 1
    // 1 2x2 -> 212
    // 1 2x2 3xx3 -> 32x*, 31x3+2x2 31232
    // 1 2x2 3xx3 4xxx4 -> 4xxx4
    //                     .3xx3*
    //                     .2x2
    //                     . 3xx3
    //                     .     1
    //                     4232431
    // 1 2x2 3xx3 4xxx4 5xxxx5 ->
    //                  5xxxx5         5
    //                  .4xxx4*
    //                  .3xx3.         3
    //                  . 4xxx4    *
    //                  .  2x2*    *
    //                  .  1 .     *
    //                  .   2x2*   *
    //                  . 2x2*
    //                  . 1  .         1
    //                  .  4xxx4       4
    //                  .     2x2      2
    //                  531435242

```
We try to place every number, and back track if it is not possible.

#### Approach

* result and number set can be the single instance or the copies
* joke hardcoded solution

#### Complexity

- Time complexity:
$$O(n^n)$$, recursion depth is `n` and each iterates over `n`

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun constructDistancedSequence(n: Int): IntArray? {
        fun dfs(i: Int, s: List<Int>, r: IntArray): IntArray? =
            if (s.size < 1) r else if (r[i] > 0) dfs(i + 1, s, r)
            else s.filter { x -> x < 2 || i + x < r.size && r[i + x] < 1 }
            .firstNotNullOfOrNull { x -> val c = r.clone(); c[i] = x;
                if (x > 1) c[i + x] = x; dfs(i + 1, s - x, c) }
        return dfs(0, (n downTo 1).toList(), IntArray(n * 2 - 1))
    }

```
```rust

    pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
        let (mut r, mut u) = (vec![0; n as usize * 2 - 1], vec![false; n as usize + 1]);
        fn dfs(i: usize, r: &mut Vec<i32>, u: &mut Vec<bool>) -> bool {
            if i == r.len() { return true }; if r[i] > 0 { return dfs(i + 1, r, u) }
            for x in (1..u.len()).rev() { if !u[x] && (x < 2 || i + x < r.len() && r[i + x] < 1) {
                u[x] = true; r[i] = x as i32; if x > 1 { r[i + x] = x as i32 };
                if dfs(i + 1, r, u) { return true }; u[x] = false; r[i] = 0; if x > 1 { r[i + x] = 0 }
            }}; false
        }; dfs(0, &mut r, &mut u); r
    }

```
```c++

    vector<int> constructDistancedSequence(int n) {
        return vector<vector<int>>{
{1},
{2,1,2},
{3,1,2,3,2},
{4,2,3,2,4,3,1},
{5,3,1,4,3,5,2,4,2},
{6,4,2,5,2,4,6,3,5,1,3},
{7,5,3,6,4,3,5,7,4,6,2,1,2},
{8,6,4,2,7,2,4,6,8,5,3,7,1,3,5},
{9,7,5,3,8,6,3,5,7,9,4,6,8,2,4,2,1},
{10,8,6,9,3,1,7,3,6,8,10,5,9,7,4,2,5,2,4},
{11,9,10,6,4,1,7,8,4,6,9,11,10,7,5,8,2,3,2,5,3},
{12,10,11,7,5,3,8,9,3,5,7,10,12,11,8,6,9,2,4,2,1,6,4},
{13,11,12,8,6,4,9,10,1,4,6,8,11,13,12,9,7,10,3,5,2,3,2,7,5},
{14,12,13,9,7,11,4,1,10,8,4,7,9,12,14,13,11,8,10,6,3,5,2,3,2,6,5},
{15,13,14,10,8,12,5,3,11,9,3,5,8,10,13,15,14,12,9,11,7,4,6,1,2,4,2,7,6},
{16,14,15,11,9,13,6,4,12,10,1,4,6,9,11,14,16,15,13,10,12,8,5,7,2,3,2,5,3,8,7},
{17,15,16,12,10,14,7,5,3,13,11,3,5,7,10,12,15,17,16,14,9,11,13,8,6,2,1,2,4,9,6,8,4},
{18,16,17,13,11,15,8,14,4,2,12,2,4,10,8,11,13,16,18,17,15,14,12,10,9,7,5,3,6,1,3,5,7,9,6},
{19,17,18,14,12,16,9,15,6,3,13,1,3,11,6,9,12,14,17,19,18,16,15,13,11,10,8,4,5,7,2,4,2,5,8,10,7},
{20,18,19,15,13,17,10,16,7,5,3,14,12,3,5,7,10,13,15,18,20,19,17,16,12,14,11,9,4,6,8,2,4,2,1,6,9,11,8}}[n - 1];}

```

