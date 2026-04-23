---
layout: leetcode-entry
title: "440. K-th Smallest in Lexicographical Order"
permalink: "/leetcode/problem/2025-06-09-440-k-th-smallest-in-lexicographical-order/"
leetcode_ui: true
entry_slug: "2025-06-09-440-k-th-smallest-in-lexicographical-order"
---

[440. K-th Smallest in Lexicographical Order](https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/description/) hard
[blog post](https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/solutions/6825462/kotlin-rust-by-samoylenkodmitry-4fb7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09062025-440-k-th-smallest-in-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vkukdbjhLNk)
![1.webp](/assets/leetcode_daily_images/3e428fe0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1014

#### Problem TLDR

k-th lexicographical number of 1..n #hard

#### Intuition

Didn't solve. (previous attempt were passed but not optimal https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/solutions/5819960/kotlin-rust/)

```j
// generate all - will give TLE, 10^9 range
    // [1, 10,
    //        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    //     11,
    //        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    //     12,
    //        120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    //     13,
    //        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    //     14,
    //        140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    //     15,
    //        150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    //     16,
    //        160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
    //     17,
    //        170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    //     18,
    //        180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    //     19,
    //        190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
    // 2, 20,
    //        200,
    //        21, 22, 23, 24, 25, 26, 27, 28, 29,
    // 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    // 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    // 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    // 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    // 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    // 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    // 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    // (8 minute) no ideas
    // ok, let's try brute-force maybe it will be fast enough
    // (15 minute) TLE n = 957747794  k = 424238336
    // from hints: look for digits in n to form a subtree
    // 1 - subtree of 1 has size of sz1 (110)
    // 2 - subtree of 2 has size of sz2 (10 + 1), 200=200, 1 extra child
    // 3 - subtree of 3 has size of sz3 (10)  300 bigger than 200, no extra
    // 4 - subtree of 4 has size of sz4 (10)
    // ...
    // 9 - subtree of 9 has size of sz9 (10)
    //
    // then remove first digit and go deeper at 1 (the first digit of solution k = 3,n=200)
    // 10 - (10)
    // 11 - (10)
    // ...
    // 19 - (10)
    // then go deeper at 10 (the second digit of solution k = 3-1=2, n=20)
    // 100 - (0) the third digit of solution, k = 2-1=1, n= 2? (how to adjust n?)
    // 101 - (0)
    // ...
    // 109 - (0)
    // (50 minute) still didn't know how to adjust n
    // 54 look for solution
```

The solution:
* start with root of the tree x = 1
* calculate subtree size from..to by doing from *= 10, to = to * 10 + 9
* if i + count > k, we found the digit, go deeper subtree x *= 10
* if i + count <= k, skip the node x++, i += count

#### Approach

* the previous approach of just stealing the solution didn't give good results as I am unable to solve it again; better spend more time to understand the logic

#### Complexity

- Time complexity:
$$O(lg(n)lg(k))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 0ms
    fun findKthNumber(n: Int, k: Int): Int {
        var x = 1L; var i = 1; var n = n.toLong(); var k = k.toLong()
        while (i < k) {
            var cnt = 0L; var from = x; var to = x
            while (from <= n) {
                cnt += min(to, n) - from + 1
                from *= 10; to = to * 10 + 9
            }
            if (i + cnt <= k) { i += cnt.toInt(); x++ } else { i++; x *= 10 }
        }
        return x.toInt()
    }

```

```rust

// 0ms
    pub fn find_kth_number(n: i32, k: i32) -> i32 {
        let (mut x, n, mut k) = (1i64, n as i64, k as i64);
        while k > 1 {
            let (mut skip, mut from, mut to) = (0, x, x);
            while from <= n {
                skip += to.min(n) - from + 1;
                from *= 10; to = to * 10 + 9
            }
            if k - skip < 1 { k -= 1; x *= 10 } else { k -= skip; x += 1 }
        } x as _
    }

```
```c++

// 0ms
    int findKthNumber(int n, int k) {
        long long x = 1; k--;
        while (k) {
            long long skip = 0, from = x, to = x;
            while (from <= n)
                skip += min(to, 1LL * n) - from + 1,
                from *= 10, to = to * 10 + 9;
            if (k - skip < 0) --k, x *= 10; else ++x, k -= skip;
        } return (int) x;
    }

```

