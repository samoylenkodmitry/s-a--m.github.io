---
layout: leetcode-entry
title: "3356. Zero Array Transformation II"
permalink: "/leetcode/problem/2025-03-13-3356-zero-array-transformation-ii/"
leetcode_ui: true
entry_slug: "2025-03-13-3356-zero-array-transformation-ii"
---

[3356. Zero Array Transformation II](https://leetcode.com/problems/zero-array-transformation-ii/description/) medium
[blog post](https://leetcode.com/problems/zero-array-transformation-ii/solutions/6531748/kotlin-rust-by-samoylenkodmitry-0wek/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13032025-3356-zero-array-transformation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aT23SQWOqXA)
![1.webp](/assets/leetcode_daily_images/48ef5797.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/926

#### Problem TLDR

Min queries (l..r, v) to decrease nums by v to 0 #medium #line_sweep

#### Intuition

Didn't solve without the hint: the binary search works here.

Let's try example:

```j

    // 0 1 2 3      0       1       2       3
    // 2 5 2 3    j 0..2 2, 1..3 1, 2..3 2, 1..1 3
    // [...]   2  0
    //   [...] 1  1
    //   []    3  3
    //     [.] 2  2

```

I've tried to do a single pass solution:
* sort the queries by left border
* on each number take all left borders and remove all right borders (I used PriorityQueue for removals)
* try to pick max k (that's where I failed, there is no way to do this on a sorted queries right)

So, just sorting didn't work. I have to resort to the hint and consider the BinarySearch.

Let's simplify the picking of `k`:
* we already know the `k` (middle of the BinarySearch range)
* drop all queries that bigger
* calculate the sum

That's worked out. However, solution no longer require the sorting, just prepare line sweep array: add v at the range start l, remove v at the range end r + 1. Much faster.

And now, the other's guy solution: didn't have to do the Binary Search at all, just do the same line sweep, and dynamically adjust current sum when increasing the k.

#### Approach

* try to solve at least in 40 minutes mark
* then look for the hints
* after 1 hour it's a fair game to steal the solution

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minZeroArray(nums: IntArray, q: Array<IntArray>): Int {
        var k = 0; var sum = 0; val s = IntArray(nums.size + 1)
        for (i in nums.indices) {
            sum += s[i]
            while (sum < nums[i]) {
                if (k >= q.size) return -1
                val (l, r, v) = q[k++]
                s[l] += v; s[r + 1] -= v
                if (i in l..r) sum += v
            }
        }
        return k
    }

```
```rust

    pub fn min_zero_array(nums: Vec<i32>, q: Vec<Vec<i32>>) -> i32 {
        let (mut k, mut sum, mut s) = (0, 0, vec![0; nums.len() + 1]);
        for i in 0..nums.len() {
            sum += s[i];
            while sum < nums[i] {
                if k >= q.len() { return -1 }
                let (l, r, v) = (q[k][0] as usize, q[k][1] as usize, q[k][2]);
                s[l] += v; s[r + 1] -= v; k += 1;
                if (l..=r).contains(&i) { sum += v }
            }
        }; k as _
    }

```
```c++

    int minZeroArray(vector<int>& nums, vector<vector<int>>& q) {
        vector<int> s(size(nums) + 1); int k = 0, sum = 0;
        for (int i = 0; i < size(nums); ++i) {
            sum += s[i];
            while (sum < nums[i]) {
                if (k >= size(q)) return -1;
                int l = q[k][0], r = q[k][1], v = q[k++][2];
                s[l] += v; s[r + 1] -= v;
                if (l <= i && i <= r) sum += v;
            }
        } return k;
    }

```

