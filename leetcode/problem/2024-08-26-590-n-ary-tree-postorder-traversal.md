---
layout: leetcode-entry
title: "590. N-ary Tree Postorder Traversal"
permalink: "/leetcode/problem/2024-08-26-590-n-ary-tree-postorder-traversal/"
leetcode_ui: true
entry_slug: "2024-08-26-590-n-ary-tree-postorder-traversal"
---

[590. N-ary Tree Postorder Traversal](https://leetcode.com/problems/n-ary-tree-postorder-traversal/description/) easy
[blog post](https://leetcode.com/problems/n-ary-tree-postorder-traversal/solutions/5691953/kotlin-c/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26082024-590-n-ary-tree-postorder?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5bYjDsxhduU)
![1.webp](/assets/leetcode_daily_images/2bdd1dae.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/714

#### Problem TLDR

Postorder tree traversal #easy #tree

#### Intuition

Visit children, then append current.

#### Approach

We can use the stack for iteration without recursion. Or we can use recursion with a separate collection to make it faster.

* let's just reuse the method's signature neglecting the performance

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun postorder(root: Node?): List<Int> = root?.run {
        children.map { postorder(it) }.flatten() + listOf(`val`)
    } ?: listOf()

```
```c++

    vector<int> postorder(Node* root) {
        vector<int> res;
        if (!root) return res;
        for (auto c : root->children)
            for (auto x : postorder(c))
                res.push_back(x);
        res.push_back(root->val);
        return res;
    }

```

