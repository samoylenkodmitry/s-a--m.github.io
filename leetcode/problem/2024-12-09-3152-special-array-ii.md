---
layout: leetcode-entry
title: "3152. Special Array II"
permalink: "/leetcode/problem/2024-12-09-3152-special-array-ii/"
leetcode_ui: true
entry_slug: "2024-12-09-3152-special-array-ii"
---

[3152. Special Array II](https://leetcode.com/problems/special-array-ii/description/) medium
[blog post](https://leetcode.com/problems/special-array-ii/solutions/6128306/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09122024-3152-special-array-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/PDLfGII2I5Q)
[deep-dive](https://notebooklm.google.com/notebook/438184b2-0566-4054-a47f-eaa68def1c9b/audio)
![1.webp](/assets/leetcode_daily_images/5200fa9e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/827

#### Problem TLDR

Queries is all adjucents parity differ [i..j] #medium #two_pointers

#### Intuition

Let's observe the data and build an intuition:

```j

    // 1 1 2 3 4 4 5 6 6 7 7
    //  0 1 1 1 0 1 1 0 1 0
    //   j   i
    //           j i
    // [0]
    // [ 0 ]
    // [  0  ]
    //     [1]   >= j = 1
    //   [ 1 ]   >= j = 1
    // [  0  ]    < j = 0
    //   [             ]
    //   [           ]
    //   [         ]
    //   [       ]
    //   [     ]
    //   [   ]

```

The interesting observations:
* we can build a parity-diff array
* for each `end` index, the `start` index should be in the same island of `1`-ones in parity diff array

We can move two pointers, one for `end` border, second `j` for the start of the `1-island`. All queries inside it would have `start >= j`.

Another, superior tactic is to enumerate all islands, giving them uniq index or key, and then just check if both `start` and `end` have the same `key`.

#### Approach

* two-pointer is more familar for those who solve too many leetcodes, but if you take a pause and think one step more you could spot the island-indexing tactic

#### Complexity

- Time complexity:
$$O(nlog(n))$$, or O(n)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun isArraySpecial(nums: IntArray, queries: Array<IntArray>): BooleanArray {
        val g = IntArray(nums.size)
        for (i in 1..<nums.size) g[i] = g[i - 1] + 1 - abs(nums[i] - nums[i - 1]) % 2
        return BooleanArray(queries.size) { g[queries[it][0]] == g[queries[it][1]] }
    }

```

```kotlin

    fun isArraySpecial(nums: IntArray, queries: Array<IntArray>): BooleanArray {
        val inds = queries.indices.sortedBy { queries[it][1] }
        var j = 0; var k = 0; val res = BooleanArray(queries.size)
        for (i in nums.indices) {
            if (i > 0 && nums[i] % 2 == nums[i - 1] % 2) j = i
            while (k < queries.size && queries[inds[k]][1] <= i)
                    res[inds[k]] = queries[inds[k++]][0] >= j
        }
        return res
    }

```

```rust

    pub fn is_array_special(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<bool> {
        let mut g = vec![0i32; nums.len()];
        for i in 1..nums.len() { g[i] = g[i - 1] + 1 - (nums[i] - nums[i - 1]).abs() % 2 }
        queries.iter().map(|q| g[q[0] as usize] == g[q[1] as usize]).collect()
    }

```

```c++

    vector<bool> isArraySpecial(vector<int>& nums, vector<vector<int>>& q) {
        vector<int> g(nums.size()); vector<bool> r(q.size());
        for (int i = 1; i < nums.size(); ++i)
            g[i] = g[i - 1] + 1 - abs(nums[i] - nums[i - 1]) % 2;
        for (int i = 0; i < q.size(); ++i)
            r[i] = g[q[i][0]] == g[q[i][1]];
        return r;
    }

```

