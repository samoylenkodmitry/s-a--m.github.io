---
layout: leetcode-entry
title: "1524. Number of Sub-arrays With Odd Sum"
permalink: "/leetcode/problem/2025-02-25-1524-number-of-sub-arrays-with-odd-sum/"
leetcode_ui: true
entry_slug: "2025-02-25-1524-number-of-sub-arrays-with-odd-sum"
---

[1524. Number of Sub-arrays With Odd Sum](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/description/) medium
[blog post](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/solutions/6465510/kotlin-rust-by-samoylenkodmitry-owh1/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25022025-1524-number-of-sub-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NxxN50DS0h0)
![1.webp](/assets/leetcode_daily_images/e4092772.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/906

#### Problem TLDR

Count odd sum subarrays #medium #prefix_sum

#### Intuition

Didn't solve without a hint.

First, how to count subarrays: append every num only once and check some condition in O(1). We can prepare prefix sums. The interesting part is how many odd/even prefix sums we have seen before:

```j

    // 1 3 5     s  o     e     cnt
    // *         1  1           +1
    //   *       4  2     1     +1
    //     *     9  4           +2

    // 1 2 3 4 5 6 7    s    o     e    cnt
    // *                1    1          +1
    //   *              3    2     1    +1
    // 1 .o
    // 1 2o  2 is even, e++, prev o = 1, prev even = 0
    //   2                   new o = [1], [1 2] = 2
    // 1 2                   new even = [2] = 1
    //     *            6
    // 1   .o
    // 1 2 .o
    // 1 2 3e  3 is odd, odd++, prev o = [1][12] = 2, e = [2]
    //     3                    new o = [1][12]+[3][23], e=[2]+[123]
    //   2 3                                    (123-12)
    // 1 2 3                                       (123-1)
    //       *
    // 1     .o
    // 1 2   .o
    // 1 2 3 .e
    // 1 2 3 4e   4 is even,  e++, prev o=[1][12][3],e=[2][123]
    //       4  s - (1 2 3)e  new o=[1][12][3][23]+[234][34]      e=[2][123]+[4][1234]
    //     3 4  s - (1 2)o                         (1234-1)
    //   2 3 4  s - (1)o                                (1234-12)

```

For example `1 2 3 4`, when we are adding the `4`, the new subarrays are: `[4], [34], [234], [1234]`; each is a concatenation of the prefix: `[123][4], [12][34], [1][234], [][1234]`. The math for odd/even is `[odd..][..odd4]=[even4], [even..][..odd4] = [odd4], [even..][..even4] = [even4]`. So, if the current prefix is `even` we are adding all previous `odd` prefixes count on vice-versa.

However, that didn't work :)

That `1 + ` cost me 1 hour and all the hints:
```j

            if (s % 2 == 0) {
                // every odd prefix is odd suffix
                o += so
                se++
            } else {
                // every even prefix is odd suffix
                o += 1 + se
                so++
            }

```
Or in the other solutions the `1` is the starting condition for `even` count. Why so? (I still don't "get" the intuition) Example: `[21]`, we are adding `1`, sum is `3 odd`, total number of odds are: `[1] and [21]`. Let's add another `2`: [212], sum `5 odd`, odds are `[1][21]` + `[12],[212]`. That's probably a `zero-condition`: the previous `even`s count is never includes the full subarray `212`, so, we have to consider it themselves with `+1`.

#### Approach

* you can understand and even get the `idea` but to solve the problem, all the corner cases must be solved; that's a deeper understanding
* sum can be stripped down to `0-1` value of `sum % 2`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numOfSubarrays(arr: IntArray): Int {
        var o = 0; var s = 0; var so = 0; var se = 0
        for (x in arr) {
            s += x; val odd = s % 2
            o = (o + (1 - odd) * so + odd * (1 + se)) % 1000000007
            se += 1 - odd; so += odd
        }
        return o
    }

```
```rust

    pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
        let (mut o, mut s, mut so, mut se) = (0, 0, 0, 0);
        for x in arr {
            s = (s + x) & 1;
            o = (o + (1 - s) * so + s * (1 + se)) % 1000000007;
            se += 1 - s; so += s
        }; o
    }

```
```c++

    int numOfSubarrays(vector<int>& arr) {
        int r = 0, s = 0, o = 0, e = 0, m = 1e9+7;
        for (int x: arr)
            s = (s + x) & 1,
            r = (r + (1 - s) * o + s * (1 + e)) % m,
            e += 1 - s, o += s;
        return r;
    }

```

