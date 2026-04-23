---
layout: leetcode-entry
title: "1530. Number of Good Leaf Nodes Pairs"
permalink: "/leetcode/problem/2024-07-18-1530-number-of-good-leaf-nodes-pairs/"
leetcode_ui: true
entry_slug: "2024-07-18-1530-number-of-good-leaf-nodes-pairs"
---

[1530. Number of Good Leaf Nodes Pairs](https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/description/) medium
[blog post](https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/solutions/5494382/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18072024-1530-number-of-good-leaf?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KhACBS_RW78)
![2024-07-18_08-49_1.webp](/assets/leetcode_daily_images/a3cc3b94.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/674

#### Problem TLDR

Count at most `distance` paths between leaves #medium #tree

#### Intuition

Let's move up from leaves and see what information we should preserve:
![2024-07-18_08-01.webp](/assets/leetcode_daily_images/a583def2.webp)
* there are at most 10 levels for the given problem set
* we should compare the `left` node levels counts with the `right` node
* we should check all levels combinations 1..10 for the left, and 1..10 for the right
* individual leaves are irrelevant, all the distances are equal to their level

#### Approach

* We can use a HashMap, or just an array.
* The `level` parameter is not required, just move one level up from the left and right results.

#### Complexity

- Time complexity:
$$O(nlog^2(n))$$

- Space complexity:
$$O(log^2(n))$$, log(n) for the call stack, and at each level we hold log(n) array of the result

#### Code

```kotlin

    fun countPairs(root: TreeNode?, distance: Int): Int {
        var res = 0
        fun dfs(n: TreeNode?): IntArray = IntArray(11).apply {
            if (n != null)
            if (n.left == null && n.right == null) this[1] = 1 else {
                val l = dfs(n.left); val r = dfs(n.right)
                for (i in 1..10) for (j in 1..distance - i) res += l[i] * r[j]
                for (i in 1..9) this[i + 1] = l[i] + r[i]
        }}
        dfs(root)
        return res
    }

```
```rust

    pub fn count_pairs(root: Option<Rc<RefCell<TreeNode>>>, distance: i32) -> i32 {
        fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, res: &mut i32, d: usize) -> Vec<i32> {
            let mut arr = vec![0; 11]; let Some(n) = n else { return arr };
            let n = n.borrow();
            if n.left.is_none() && n.right.is_none() { arr[1] = 1 } else {
                let l = dfs(&n.left, res, d); let r = dfs(&n.right, res, d);
                for i in 1..11 { for j in 1..11 { if i + j <= d { *res += l[i] * r[j] }}}
                for i in 1..10 { arr[i + 1] = l[i] + r[i] }
            }; arr
        }
        let mut res = 0; dfs(&root, &mut res, distance as usize); res
    }

```

